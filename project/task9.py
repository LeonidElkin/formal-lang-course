from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Set, Tuple, TypeVar
from pyformlang.finite_automaton import Symbol
import networkx as nx
from pyformlang import rsa

GSSNodeT = TypeVar("GSSNodeT", bound="GSSNode")


@dataclass(frozen=True)
class RsmState:
    var: Symbol
    sub_state: str


@dataclass(frozen=True)
class SPPFNode:
    gssn: GSSNodeT
    state: RsmState
    node: int


@dataclass
class RsmStateData:
    term_edges: Dict[Symbol, RsmState] = field(default_factory=dict)
    var_edges: Dict[Symbol, Tuple[RsmState, RsmState]] = field(default_factory=dict)
    is_final: bool = False


class GSSNode:
    def __init__(self, state: RsmState, node: int):
        self.state = state
        self.node = node
        self.edges: Dict[RsmState, Set[GSSNode]] = defaultdict(set)
        self.pop_set: Set[int] = set()

    def pop(self, cur_node: int) -> Set[SPPFNode]:
        if cur_node in self.pop_set:
            return set()

        self.pop_set.add(cur_node)
        return {
            SPPFNode(gss_node, new_state, cur_node)
            for new_state, targets in self.edges.items()
            for gss_node in targets
        }

    def add_edge(self, ret_state: RsmState, target: GSSNodeT) -> Set[SPPFNode]:
        new_nodes = set()
        if target not in self.edges[ret_state]:
            self.edges[ret_state].add(target)
            new_nodes = {SPPFNode(target, ret_state, node) for node in self.pop_set}
        return new_nodes


class GllCFPQSolver:
    def __init__(self, rsm: rsa.RecursiveAutomaton, graph: nx.DiGraph):
        self.rsmstate2data = self.__init_rsm_data(rsm)
        self.start_rstate = RsmState(
            rsm.initial_label, rsm.boxes[rsm.initial_label].dfa.start_state.value
        )
        self.unprocessed: Set[SPPFNode] = set()
        self.added: Set[SPPFNode] = set()

        self.gss = {}
        self.accept_gssnode = self.__get_node(RsmState(Symbol("$"), "fin"), -1)

        self.nodes2edges = defaultdict(lambda: defaultdict(set))

        for from_node, to_node, symb in graph.edges(data="label"):
            if symb is not None:
                self.nodes2edges[from_node][symb].add(to_node)

    def __get_node(self, rsm_state: RsmState, node: int) -> GSSNode:
        return self.gss.setdefault((rsm_state, node), GSSNode(rsm_state, node))

    @staticmethod
    def __init_rsm_data(rsm: rsa.RecursiveAutomaton):
        rsmstate2data = defaultdict(dict)
        for var, box in rsm.boxes.items():
            gbox = box.dfa.to_networkx()
            for sub_state in gbox.nodes:
                rsmstate2data[var][sub_state] = RsmStateData(
                    {}, {}, sub_state in box.dfa.final_states
                )
            for from_st, to_st, symb in gbox.edges(data="label"):
                if symb is not None:
                    if Symbol(symb) not in rsm.boxes:
                        rsmstate2data[var][from_st].term_edges[symb] = RsmState(
                            var, to_st
                        )
                    else:
                        target_box = rsm.boxes[Symbol(symb)]
                        rsmstate2data[var][from_st].var_edges[symb] = (
                            RsmState(Symbol(symb), target_box.dfa.start_state.value),
                            RsmState(var, to_st),
                        )
        return rsmstate2data

    def __add_nodes(self, snodes: Set[SPPFNode]):
        new_nodes = snodes - self.added
        self.added |= new_nodes
        self.unprocessed |= new_nodes

    def __filter_nodes(
        self, snodes: Set[SPPFNode], prev_snode: SPPFNode
    ) -> Tuple[Set[SPPFNode], Set[Tuple[int, int]]]:
        node_res_set = set()
        start_fin_res_set = set()

        for sn in snodes:
            if sn.gssn == self.accept_gssnode:
                start_node = prev_snode.gssn.node
                fin_node = sn.node
                start_fin_res_set.add((start_node, fin_node))
            else:
                node_res_set.add(sn)

        return node_res_set, start_fin_res_set

    def __step(self, sppfnode: SPPFNode) -> Set[Tuple[int, int]]:
        rsm_dat = self.rsmstate2data[sppfnode.state.var][sppfnode.state.sub_state]
        res_set = set()

        graph_terms = self.nodes2edges.get(sppfnode.node, {})
        new_sppf_nodes = {
            SPPFNode(sppfnode.gssn, rsm_dat.term_edges[term], gn)
            for term, rsm_new_st in rsm_dat.term_edges.items()
            if term in graph_terms
            for gn in graph_terms[term]
        }
        self.__add_nodes(new_sppf_nodes)

        for var_start_rsm_st, ret_rsm_st in rsm_dat.var_edges.values():
            inner_gss_node = self.__get_node(var_start_rsm_st, sppfnode.node)
            post_pop_sppf_nodes = inner_gss_node.add_edge(ret_rsm_st, sppfnode.gssn)

            filtered_sppf_nodes, sub_start_fin_set = self.__filter_nodes(
                post_pop_sppf_nodes, sppfnode
            )
            self.__add_nodes(filtered_sppf_nodes)
            self.__add_nodes(
                {SPPFNode(inner_gss_node, var_start_rsm_st, sppfnode.node)}
            )
            res_set |= sub_start_fin_set

        if rsm_dat.is_final:
            pop_nodes = sppfnode.gssn.pop(sppfnode.node)
            filtered_pop_nodes, final_start_fin_set = self.__filter_nodes(
                pop_nodes, sppfnode
            )
            self.__add_nodes(filtered_pop_nodes)
            res_set |= final_start_fin_set

        return res_set

    def __call__(
        self,
        from_nodes: Set[int],
        to_nodes: Set[int],
    ) -> Set[Tuple[int, int]]:
        reach_set = set()
        for snode in from_nodes:
            gssn = self.__get_node(self.start_rstate, snode)
            gssn.add_edge(RsmState(Symbol("$"), "fin"), self.accept_gssnode)

            self.__add_nodes({SPPFNode(gssn, self.start_rstate, snode)})

        while self.unprocessed:
            reach_set |= self.__step(self.unprocessed.pop())

        filtered_set = {st_fin for st_fin in reach_set if st_fin[1] in to_nodes}
        return filtered_set


def gll_based_cfpq(
    rsm: rsa.RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: Set[int] | None = None,
    final_nodes: Set[int] | None = None,
) -> Set[Tuple[int, int]]:
    return GllCFPQSolver(rsm, graph)(
        start_nodes if start_nodes else graph.nodes(),
        final_nodes if final_nodes else graph.nodes(),
    )
