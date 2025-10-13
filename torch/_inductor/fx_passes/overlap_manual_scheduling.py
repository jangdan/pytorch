import itertools
import re
import heapq
from itertools import count
from collections import Counter, defaultdict, OrderedDict

import torch
import torch.fx as fx
from torch._dynamo.graph_deduplication import _stable_topological_sort
from torch._inductor.fx_passes.bucketing import (
    is_all_gather_into_tensor as is_all_gather,
    is_reduce_scatter_tensor as is_reduce_scatter,
    is_wait_tensor,
    merge_all_gather_bucket,
    merge_reduce_scatter_bucket,
)
from torch._inductor.fx_passes.overlap_preserving_bucketer import (
    bucket_key,
    OverlapPreservingBucketer,
)
from torch._inductor.fx_passes.overlap_scheduling import OverlapScheduler, is_compute_node, CollectiveInfo
from torch.utils._ordered_set import OrderedSet


def _get_module_stack(node):
    if node.meta.get("nn_module_stack", "") == "":
        if node.meta.get("fwd_nn_module_stack", "") != "":
            return list(node.meta["fwd_nn_module_stack"].values())
        return []
    return list(node.meta["nn_module_stack"].values())


def _addindent(s_, num_spaces):
    s = s_.split("\n")
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(num_spaces * " ") + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s


class Container:
    def __init__(self, name, klass):
        self.name = name
        self.klass = klass
        self.data = []
        self.unique_nodes = OrderedSet()
        # self.children = defaultdict(Container)
        self.children = {}

    def add(self, data):
        if data not in self.unique_nodes:
            self.data.append(data)
            self.unique_nodes.add(data)

    def get_child(self, module_stack, klass=None):
        if module_stack not in self.children:
            new_stack = Container(module_stack, klass)
            self.children[module_stack] = new_stack
        return self.children[module_stack]

    def __getitem__(self, name):
        return self.children[name]

    def __getattr__(self, name):
        return self.children[name]

    def __repr__(self):
        child_lines = []
        for name, child in self.children.items():
            mod_str = repr(child)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + name + "): " + mod_str)
        main_str = self.klass.__name__ + "("
        if child_lines:
            main_str += "\n  " + "\n  ".join(child_lines) + "\n"
        main_str += ")"
        return main_str

    def graph_view(self):
        return _make_subgraph(self.data)


def _clean_stack_name(val):
    # TODO: is this still needed?
    name = (
        val.replace("L['self']", "Model")
        .replace("_modules['", "")
        .replace("['", ".")
        .replace("']", "")
    )
    return name


def _find_key_nodes(nodes):
    root = []
    outputs = []
    nodes_set = OrderedSet(nodes)
    for node in nodes:
        for x in node.all_input_nodes:
            if x not in nodes_set:
                root.append(x)
        if all(x not in nodes_set for x in node.users):
            outputs.append(node)
    return root, outputs


def _make_subgraph(nodes):
    placeholders, outputs = _find_key_nodes(nodes)

    new_graph = torch.fx.Graph()
    env = {}

    # pyre-ignore
    def env_lookup(x: torch.fx.Node) -> torch.fx.Node:
        assert x in env, f"Dependent node {x} not in env when creating downstream node"
        return env[x]

    # pyre-ignore
    def node_copy(node, arg_transform) -> torch.fx.Node:
        if node not in env:
            new_node = new_graph.node_copy(node, arg_transform=arg_transform)
            env[node] = new_node
        else:
            new_node = env[node]
        return new_node

    for node in placeholders:
        env[node] = new_graph.placeholder(node.name)

    for node in nodes:
        if node in placeholders:
            continue
        else:
            new_node = node_copy(node, env_lookup)
            new_node.meta = node.meta.copy()

    out_node = [env[x] for x in outputs]
    new_graph.output(out_node)
    return new_graph


def _is_root(stack):
    return stack == ""


def make_graph_view(graph):
    """
    Code from: https://github.com/meta-pytorch/autoparallel/pull/158

    Make a graph view from the fx.Graph. This is a tree structure that
    represents the module hierarchy of the graph, and enables us to
    easily find the nodes that belong to each module, and gives a slightly
    easier way of visualize different parts of the graph by extracting
    subgraphs that belong to a particular module FQN.

    For example, if we have the following model with module hierarchy:

    Transformer(
        (tok_embeddings): Embedding(128256, 4096)
        (layers): ModuleDict(
            (0): TransformerBlock(
            (attention): Attention(
                (wq): Linear(in_features=4096, out_features=4096, bias=False)
                (wk): Linear(in_features=4096, out_features=1024, bias=False)
                (wv): Linear(in_features=4096, out_features=1024, bias=False)
                (wo): Linear(in_features=4096, out_features=4096, bias=False)
                (sdpa): ScaledDotProductAttention()
            )
            (feed_forward): FeedForward(
                (w1): Linear(in_features=4096, out_features=14336, bias=False)
                (w2): Linear(in_features=14336, out_features=4096, bias=False)
                (w3): Linear(in_features=4096, out_features=14336, bias=False)
            )
            (attention_norm): RMSNorm((4096,), eps=1e-05, elementwise_affine=True)
            (ffn_norm): RMSNorm((4096,), eps=1e-05, elementwise_affine=True)
            )
        )
        (norm): RMSNorm((4096,), eps=1e-05, elementwise_affine=True)
        (output): Linear(in_features=4096, out_features=128256, bias=False)
    )

    Then we can get a GraphView for the fx.Graph that enables us to do

    >>> graph_view = make_graph_view(graph)
    >>> subgraph = graph_view.layers["0"].attention.graph_view()

    where subgraph is a fx.Graph that contains all the nodes that belong to
    Transformer.layers['0'].attention, and whose inputs are all inputs to this
    region of the graph, and whose outputs are all outputs of this region of
    the graph. This returns a new graph with new nodes, so we shouldn't use it
    for graph manipulations, but it is useful to visualize what a particular
    part of a larger graph looks like.

    Additionally, you can also query the original nodes in that region with
    `graph_view.layers['0'].attention.data`, which returns a list of all the
    nodes that belong to Transformer.layers['0'].attention.
    """
    nodes = list(graph.nodes)
    nodes_by_module_stack_root = None
    last_module_name = None
    for node in nodes:
        # TODO: handle cases where there is no module stack (i.e., loop is empty and node is not added)
        for module_stack, module_class in _get_module_stack(node):
            module_stack = _clean_stack_name(module_stack)
            nodes_by_module_stack = nodes_by_module_stack_root
            for name in module_stack.split("."):
                if nodes_by_module_stack is None:
                    nodes_by_module_stack = Container(name, module_class)
                    nodes_by_module_stack_root = nodes_by_module_stack
                if _is_root(module_stack):
                    new_stack = nodes_by_module_stack
                else:
                    new_stack = nodes_by_module_stack.get_child(name, module_class)
                nodes_by_module_stack = new_stack
                nodes_by_module_stack.add(node)

    return nodes_by_module_stack_root


def decode_module(bucketing_plan: list[str]) -> list[str]:
    """
    Convert user defined FQNs to the actual list of manuals to perform bucketing.
    """
    full_plan = []
    for module_name in bucketing_plan:
        if "+" in module_name:
            full_plan.append(module_name.split("+"))
            continue
        match = re.search(r"\[(\d+)-(\d+)\]", module_name)
        if not match:
            full_plan.append(module_name)
        else:
            start, end = map(int, match.groups())
            prefix = module_name[: match.start()]
            suffix = module_name[match.end() :]
            full_plan.extend([f"{prefix}{i}{suffix}" for i in range(start, end + 1)])
    return full_plan


def get_subgraph_by_path(graph_view, paths):
    """
    Get subgraph by path(s).
    Args:
        graph_view (object): Root graph view object.
        paths (str or list of str): Path(s) to subgraph.
    Returns:
        object: Subgraph object or node.
    """

    def get_node_by_path(node, path):
        for p in path.split("."):
            if p in node.children:
                node = node.children[p]
            else:
                return Container("", "")
        return node

    if isinstance(paths, list):
        nodes = list(
            itertools.chain.from_iterable(
                get_node_by_path(graph_view, p).data for p in paths
            )
        )
        return nodes
    else:
        node = get_node_by_path(graph_view, paths)
        return node.data


class ManualOverlapPreservingBucketer(OverlapPreservingBucketer):
    def __init__(self, node_users, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.node_users = node_users
        self.bucket_all_gather_dict: dict[int, OrderedSet[fx.Node]] = defaultdict(OrderedSet)
        self.counter = count()
        self.wait_to_node_map: dict[fx.Node, fx.Node] = defaultdict()

    def bucket_collectives(self, nodes) -> None:
        """
        Manually bucket all rs/ag/wait nodes from self.manual_nodes into one bucket.
        """

        # Filter out valid collective *starts*
        collectives = [n for n in nodes if n in self.collective_info]
        if collectives == []:
            return

        grouped_collectives: dict[object, OrderedSet[fx.Node]] = defaultdict(OrderedSet)

        for node in collectives:
            key = bucket_key(node)
            node_ancestors = self.node_ancestors[node]
            node_users = self.node_users[self.collective_info[node].wait_node]
            # We only want to bucket collectives that
            # 1. all_gather that do not have ancestors dependency with other collectives
            # 2. reduce scatter that the wait user node is returned as output
            if is_all_gather(node) and bool(
                OrderedSet(node_ancestors) & self.collective_info.keys()
            ):
                continue
            if is_reduce_scatter(node) and not next(iter(node_users)).op == "output":
                continue
            if key is not None:
                grouped_collectives[key].add(node)

        counter_idx = next(self.counter)

        def _bucket_group(coll_nodes: list[fx.Node]):
            if not coll_nodes:
                return {}, []

            waits = [self.collective_info[n].wait_node for n in coll_nodes]
            # Use earliest wait insertion point
            first_wait = min(waits, key=lambda w: self.node_idx[w])
            # Find insertion location
            first = coll_nodes[0]
            next_node = first
            while next_node in coll_nodes:
                next_node = next_node.next

            if is_all_gather(first):
                new_nodes, replacements = merge_all_gather_bucket(
                    self.graph,
                    coll_nodes,
                    wait_insertion_point=first_wait,
                    insert_before=next_node,
                    mode="custom_ops",
                )
            else:
                new_nodes, replacements = merge_reduce_scatter_bucket(
                    self.graph,
                    coll_nodes,
                    wait_insertion_point=first_wait,
                    insert_before=next_node,
                    mode="custom_ops",
                )

            # Identify the new wait and start
            new_waits = [n for n in new_nodes if is_wait_tensor(n)]
            assert len(new_waits) == 1, (
                f"Expected exactly one new wait, got {new_waits}"
            )
            new_wait = new_waits[0]
            new_start = new_wait.args[0]
            assert isinstance(new_start, fx.Node)

            node_type = "bucketed_all_gather" if is_all_gather(first) else "bucketed_reduce_scatter"
            for n in new_nodes:
                n.meta["nn_module_stack"] = coll_nodes[0].meta.get("nn_module_stack", "")
                n.meta["fwd_nn_module_stack"] = coll_nodes[0].meta.get("fwd_nn_module_stack", "")
                if n == new_wait:
                    node_type = node_type + "_wait"
                n.meta["manual_bucket_node_type"] = node_type
                if node_type == "bucketed_all_gather":
                    self.bucket_all_gather_dict[counter_idx].add(n)
                if "wait" in node_type:
                    self.wait_to_node_map[n] = new_wait

        for key, nodes in grouped_collectives.items():
            _bucket_group(list(nodes))



class ManualOverlapScheduler(OverlapScheduler):
    """
    Scheduler that manual reorders and buckets collective nodes based on user specifications (nodes_in_subgraph)
    """

    def __init__(self, gm: fx.GraphModule, buckted_module: list[str]):
        super().__init__(gm)  # use default parameters from OverlapScheduler
        # self.ready: list[tuple[object, fx.Node]] = []
        # for node in self.nodes:
        #     if self.in_degree[node] == 0:
        #         heapq.heappush(self.ready, (1, node)) # (node_depth, node)

        self.buckted_module = buckted_module
        self.nodes_in_subgraph: list[list[fx.Node]] = []

        self.bucketed_coll_nodes_in_subgraph: list[list[fx.Node]] = []
        self.node_users: dict[fx.Node, OrderedSet[fx.Node]] = self._collect_node_users()
        self.bucketer = ManualOverlapPreservingBucketer(
            graph=self.graph,
            collective_info=self.collective_info,
            node_ancestors=self.node_ancestors,
            node_users=self.node_users,
            scheduled=OrderedSet(self.graph.nodes),
        )

    def run(self) -> torch.fx.GraphModule:
        """Run the manual scheduling from nodes_in_subgraph"""
        # Bucket collective
        if torch._inductor.config.test_configs.aten_fx_overlap_preserving_bucketing:
            self._manual_bucket_collectives()
        elif torch._inductor.config.test_configs.aten_fx_overlap_insert_overlap_deps:
            # If not bucketing, add effect tokens to preserve hiding dependencies
            self._add_effect_tokens_for_overlap()

        # Reorder collectives
        self._manual_reorder_graph()

        return self.gm

    def _manual_reorder_graph(self) -> None:
        """
        Enforce:
        - all_gather_start_i depends on all_gather_wait_(i-1)
        - reduce_scatter_wait_i must happen before reduce_scatter_start_(i+1)
        """
        delayed_rs_nodes = []
        delayed_ag_nodes = []
        overlap_deps: dict[fx.Node, OrderedSet[fx.Node]] = defaultdict(OrderedSet)

        # schedule reduce scatter normally in the list
        while self.ready:
            _, node = heapq.heappop(self.ready)
            node_type = node.meta.get("manual_bucket_node_type", "")

            if node in self.scheduled:
                continue

            if node_type == "bucketed_reduce_scatter":
                for delayed in delayed_rs_nodes:
                    self._schedule(delayed)
                    overlap_deps[delayed].add(node)
                delayed_rs_nodes.clear()
                self._schedule(node)
                continue

            if node_type == "bucketed_reduce_scatter_wait":
                delayed_rs_nodes.append(node)
                continue

            self._schedule(node)

        for delayed in delayed_rs_nodes:
            self._schedule(delayed)

        # for delayed in delayed_ag_nodes:
        #     self._schedule(delayed)

        self.scheduled = list(self.scheduled)

        self.scheduled.reverse()
        picked_ag = []
        test_list = []
        last_comp_node = None

        for n in self.scheduled:
            node_type = n.meta.get("manual_bucket_node_type", "")
            if node_type == "bucketed_all_gather":
                picked_ag.append(n)
                continue
            if node_type == "bucketed_all_gather_wait":
                if len(picked_ag) > 0:
                    for ag in picked_ag:
                        overlap_deps[self.bucketer.wait_to_node_map[n]].add(ag)
                test_list.extend(picked_ag)
                picked_ag = []
            test_list.append(n)
            if is_compute_node(n):
                last_comp_node = n
        if len(delayed_rs_nodes) > 0:
            for ag in picked_ag:
                overlap_deps[last_comp_node].add(ag)
        test_list.extend(picked_ag)
        test_list.reverse()
        self.scheduled = OrderedSet(test_list)

        output_node = self.graph.output_node()
        for node in self.scheduled:
            if node.op == "placeholder":
                continue
            output_node.prepend(node)

        _stable_topological_sort(self.graph, overlap_deps)
        self.graph.lint()

    def _manual_bucket_collectives(self) -> None:
        self._obtain_nodes_in_subgraph()
        for i, nodes in enumerate(self.nodes_in_subgraph):
            self.bucketer.bucket_collectives(nodes=nodes)

        _stable_topological_sort(self.graph, {})
        self.graph.lint()
        self.nodes = list(self.graph.nodes)
        self.in_degree = Counter(user for node in self.nodes for user in node.users)

    def _collect_node_users(self) -> dict[fx.Node, OrderedSet[fx.Node]]:
        """Collect all users for each node."""
        node_users: dict[fx.Node, OrderedSet[fx.Node]] = defaultdict(OrderedSet)
        for node in self.nodes:
            for output_node in list(node.users.keys()):
                node_users[node].add(output_node)
                node_users[node] |= node_users[output_node]
        return node_users

    def _schedule(self, node: fx.Node) -> None:
        """Schedule a node."""
        assert node not in self.scheduled
        assert all(n in self.scheduled for n in node.all_input_nodes)
        self.scheduled.add(node)
        for user in node.users:
            self.in_degree[user] -= 1
            if self.in_degree[user] == 0:
                heapq.heappush(self.ready, ((), user))

    def _obtain_nodes_in_subgraph(self):
        """
        Obtain nodes in each subgraph from nodes_in_subgraph
        """
        graph_view = make_graph_view(self.graph)
        self.nodes_in_subgraph = []
        for module in self.buckted_module:
            subgraph_view = get_subgraph_by_path(graph_view.children["Model"], module)
            self.nodes_in_subgraph.append(subgraph_view)



def manual_overlap_bucketing(
    gm: torch.fx.GraphModule,
    buckted_module: list[str],
) -> torch.fx.GraphModule:
    """Schedule nodes based on user specifications in buckted_module

    Args:
        gm: Input graph module to optimize.
        buckted_module: user specified plans
    """
    ## Step 1: Get the subgraphs from each bucketed module
    buckted_module = decode_module(buckted_module)

    ## Step 2: Do bucketing and reordering over each subgraph
    return ManualOverlapScheduler(gm, buckted_module).run().recompile()
