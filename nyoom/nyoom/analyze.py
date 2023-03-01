import queue
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
from networkx.relabel import relabel_nodes
import json


class Analyze:
    def __init__(self, relname_reltuples_map: dict, indexname_tablename_map: dict, explain_dump: str):
        self._relname_reltuples_map = relname_reltuples_map
        self._indexname_tablename_map = indexname_tablename_map
        self._json = json.loads(explain_dump)
        self._graph = self._build_graph(self._json)
        self._dict = self._build_dict(self._json)
        self._pipelines = self._compute_pipelines()
        self._drivers = self._compute_drivers()
        self._bounds = self._compute_bounds()

    def _get_node(self, plan_node_id):
        return self._dict[plan_node_id]

    def _get_child(self, plan_node_id, parent_relationship):
        node = self._dict[plan_node_id]
        for child_node_id in node["Children Plan Node Ids"]:
            candidate = self._dict[child_node_id]
            if candidate["Parent Relationship"] == parent_relationship:
                return child_node_id
        raise RuntimeError

    def _get_only_child(self, plan_node_id):
        node = self._dict[plan_node_id]
        children = node["Children Plan Node Ids"]
        assert len(children) == 1
        return self._dict[children[0]]

    def _compute_pipelines(self):
        next_pipeline_id = 1
        pipelines = {}
        leaves = [n for n in self._graph.nodes() if self._graph.out_degree(n) == 0]
        workq = queue.Queue()
        for leaf in leaves:
            workq.put(leaf)

        # Tuple of plan node IDs whose pipelines should be merged.
        # Will merge into 0th index.
        mergers = []

        while not workq.empty():
            leaf = workq.get()
            plan_node_id = leaf
            pipeline_id = next_pipeline_id
            next_pipeline_id += 1
            pipelines[plan_node_id] = pipeline_id

            while True:
                node = self._dict[plan_node_id]
                node_type = node["Node Type"]
                parent_plan_node_id = node["Parent Plan Node Id"]
                if parent_plan_node_id is None:
                    break

                parent_node = self._dict[parent_plan_node_id]
                parent_type = parent_node["Node Type"]

                blocking = [
                    "Aggregate",
                    "Gather",
                    "Gather Merge",
                    "Materialize",
                    "Sort",
                ]
                nonblocking = [
                    "Bitmap Heap Scan",
                    "Hash",  # TODO(WAN): this matches our discussions and pg-progress, but odd...
                    "Index Only Scan",
                    "Index Scan",
                    "Limit",
                    "Subquery Scan",
                ]

                if parent_type in blocking:
                    workq.put(parent_plan_node_id)
                    break
                elif self._graph.out_degree(parent_plan_node_id) > 1:
                    if parent_type == "Hash Join" and node["Parent Relationship"] == "Outer":
                        # > For a Hash Join, the join operator is included in the pipeline of the probe child,
                        # > and the build child is the root of another pipeline.
                        pass
                    elif parent_type == "Merge Join" and node["Parent Relationship"] == "Outer":
                        # > For a Merge-Join, the pipelines containing its children and the Merge Join
                        # > operator itself are union'ed to create a single pipeline.
                        mergers.append((plan_node_id, parent_plan_node_id,
                                        *self._get_node(parent_plan_node_id)["Children Plan Node Ids"]))
                    elif parent_type == "Nested Loop" and node["Parent Relationship"] == "Outer":
                        # > For a Nested Loops or Index Nested Loops Join operator,
                        # > the outer child, the join operator and its entire inner subtree
                        # > are part of a [sic] the same pipeline as the outer child node.
                        inner_plan_node_id = self._get_child(parent_plan_node_id, "Inner")
                        mergers.append((plan_node_id, parent_plan_node_id, inner_plan_node_id))
                    else:
                        break
                else:
                    assert parent_type in blocking + nonblocking, f"{parent_type} unknown."
                pipelines[parent_plan_node_id] = pipeline_id
                plan_node_id = parent_plan_node_id

        uf = {i: i for i in range(next_pipeline_id)}

        def _find(_pipeline_id):
            if uf[_pipeline_id] == _pipeline_id:
                return _pipeline_id
            return _find(uf[_pipeline_id])

        def _union(pipeline_id_1, pipeline_id_2):
            x = _find(pipeline_id_1)
            y = _find(pipeline_id_2)
            uf[x] = y

        rewrite = {}
        for merge_dest_plan_node, *merge_src_plan_nodes in mergers:
            target_pipeline = pipelines[merge_dest_plan_node]
            for merging_plan_node in merge_src_plan_nodes:
                source_pipeline_id = pipelines[merging_plan_node]
                _union(target_pipeline, source_pipeline_id)

        pipelines = {plan_node_id: _find(pipeline_id) for plan_node_id, pipeline_id in pipelines.items()}

        assert len(pipelines) == len(self._dict), "Need more pipeline logic."
        return pipelines

    def _compute_drivers(self):
        leaves = [n for n in self._graph.nodes() if self._graph.out_degree(n) == 0]
        workq = queue.Queue()
        for leaf in leaves:
            workq.put(leaf)

        drivers = set()
        while not workq.empty():
            leaf = workq.get()
            plan_node_id = leaf
            drivers.add(plan_node_id)

            new_pipeline = False
            while True:
                pipeline_id = self._pipelines[plan_node_id]
                node = self._dict[plan_node_id]
                if new_pipeline:
                    drivers.add(plan_node_id)
                    new_pipeline = False
                parent_plan_node_id = node["Parent Plan Node Id"]
                if parent_plan_node_id is None:
                    break
                parent_pipeline_id = self._pipelines[parent_plan_node_id]
                if pipeline_id != parent_pipeline_id:
                    children = self._dict[parent_plan_node_id]["Children Plan Node Ids"]
                    any_match_parent = any(self._pipelines[child] == parent_pipeline_id for child in children)
                    if not any_match_parent:
                        new_pipeline = True
                plan_node_id = parent_plan_node_id

        return list(drivers)

    def _compute_bounds(self):
        workq = queue.Queue()
        leaves = [n for n in self._graph.nodes() if self._graph.out_degree(n) == 0]
        for leaf in leaves:
            workq.put(leaf)

        bounds = {
            plan_node_id: {"min": 0, "max": float("inf")}
            for plan_node_id in self._dict
        }

        while not workq.empty():
            plan_node_id = workq.get()
            while True:
                node = self._dict[plan_node_id]
                node_type = node["Node Type"]

                if node_type in ["Bitmap Heap Scan", "Bitmap Index Scan", "Index Scan", "Index Only Scan",
                                 "Sample Scan", "Seq Scan"]:
                    if node_type == "Bitmap Index Scan":
                        relname = self._indexname_tablename_map[node["Index Name"]]
                    else:
                        relname = node["Relation Name"]
                    num_tuples = self._relname_reltuples_map[relname]
                    if any([x in node for x in ["Filter", "Index Cond", "Recheck Cond"]]):
                        bounds[plan_node_id]["min"] = node["Tuples Processed"]
                        bounds[plan_node_id]["max"] = num_tuples
                    else:
                        bounds[plan_node_id]["min"] = num_tuples
                        bounds[plan_node_id]["max"] = num_tuples
                elif node_type == "Sort":
                    child = self._get_node(self._get_child(plan_node_id, "Outer"))
                    bounds[plan_node_id]["min"] = child["Tuples Processed"]
                    bounds[plan_node_id]["max"] = bounds[child["Plan Node Id"]]["max"]
                elif node_type == "Aggregate":
                    child = self._get_only_child(plan_node_id)
                    bounds[plan_node_id]["min"] = max(1, node["Tuples Processed"])
                    bounds[plan_node_id]["max"] = bounds[child["Plan Node Id"]]["max"] - max(1,
                                                                                             node["Tuples Processed"])
                elif node_type in ["Hash Join", "Merge Join", "Nested Loop"]:
                    outer = self._get_node(self._get_child(plan_node_id, "Outer"))
                    inner = self._get_node(self._get_child(plan_node_id, "Inner"))
                    bounds[plan_node_id]["min"] = node["Tuples Processed"]
                    bounds[plan_node_id]["max"] = (bounds[outer["Plan Node Id"]]["max"] - outer["Tuples Processed"]) * \
                                                  bounds[inner["Plan Node Id"]]["max"] + node["Tuples Processed"]
                elif node_type in ["Hash"]:
                    child = self._get_only_child(plan_node_id)
                    bounds[plan_node_id]["min"] = node["Tuples Processed"]
                    bounds[plan_node_id]["max"] = bounds[child["Plan Node Id"]]["max"]
                elif node_type == "Limit":
                    child = self._get_only_child(plan_node_id)
                    bounds[plan_node_id]["min"] = child["Tuples Processed"]
                    bounds[plan_node_id]["max"] = min(node["Plan Rows"], bounds[child["Plan Node Id"]]["max"])
                elif node_type in ["Gather", "Gather Merge"]:
                    # TODO(WAN): do I know what I'm doing?
                    n_workers = node["Workers Planned"]
                    child = self._get_only_child(plan_node_id)
                    bounds[plan_node_id]["min"] = bounds[child["Plan Node Id"]]["min"] * n_workers
                    bounds[plan_node_id]["max"] = bounds[child["Plan Node Id"]]["max"] * n_workers
                elif node_type == "Subquery Scan":
                    child = self._get_only_child(plan_node_id)
                    bounds[plan_node_id]["min"] = node["Tuples Processed"]
                    bounds[plan_node_id]["max"] = bounds[child["Plan Node Id"]]["max"]
                elif node_type == "Materialize":
                    bounds[plan_node_id]["min"] = node["Tuples Processed"]
                    # TODO(WAN): ?!
                    if node["Parent Relationship"] == "Inner":
                        child = self._get_only_child(plan_node_id)
                        parent = self._get_node(node["Parent Plan Node Id"])
                        if parent["Node Type"] == "Nested Loop":
                            outer = self._get_node(self._get_child(parent["Plan Node Id"], "Outer"))
                            bounds[plan_node_id]["max"] = bounds[outer["Plan Node Id"]]["max"] * \
                                                          bounds[child["Plan Node Id"]]["max"]

                parent_plan_node_id = node["Parent Plan Node Id"]
                if parent_plan_node_id is None:
                    break
                plan_node_id = parent_plan_node_id

        return bounds

    def get_victims(self, cutoff_pct=10, min_processed=1000):
        victims = []
        for driver in self._drivers:
            node = self._dict[driver]
            processed = node["Tuples Processed"]
            estimated_total = node["Plan Rows"]
            progress = processed / estimated_total * 100
            if progress < cutoff_pct:
                # print(f"Waiting for {driver} ({processed} / {estimated_total}, {progress}%)")
                pass
            elif processed >= min_processed:
                victims.append(driver)
        return victims

    @staticmethod
    def _build_graph(explain_json):
        result = nx.DiGraph()

        def _build(plan, parent):
            plan_node_id = plan["Plan Node Id"]
            result.add_node(plan_node_id)
            if parent is not None:
                result.add_edge(parent["Plan Node Id"], plan_node_id)
            for child in plan.get("Plans", []):
                _build(child, parent=plan)

        _build(explain_json["Plan"], parent=None)
        return result

    @staticmethod
    def _build_dict(explain_json):
        result = {}

        def _build(plan, parent):
            plan_node_id = plan["Plan Node Id"]
            result[plan_node_id] = {k: v for k, v in plan.items() if k != "Plans"}
            result[plan_node_id]["Children Plan Node Ids"] = [child_plan["Plan Node Id"] for child_plan in
                                                              plan.get("Plans", [])]
            result[plan_node_id]["Parent Plan Node Id"] = parent["Plan Node Id"] if parent is not None else None
            for child in plan.get("Plans", []):
                _build(child, parent=plan)

        _build(explain_json["Plan"], parent=None)
        return result

    def viz(self, filepath):
        def get_label(plan_node_id):
            node = self._dict[plan_node_id]
            node_type = node["Node Type"]
            pipeline_id = self._pipelines.get(plan_node_id, "INVALID")
            tuples_processed = node["Tuples Processed"]
            tuples_total_estimate = node["Plan Rows"]
            progress_estimate = round((tuples_processed / tuples_total_estimate) * 100, 2)
            bounds_min = self._bounds[plan_node_id]["min"]
            bounds_max = self._bounds[plan_node_id]["max"]
            bounds_progress_estimate = round((tuples_processed / bounds_max) * 100, 2)

            str_node = f"{plan_node_id} ({node['Node Type']})"
            str_pipeline = f"\nPipeline: {pipeline_id}\tDriver: {plan_node_id in self._drivers}"
            str_tuples = f"\nTuples: {tuples_processed} / {tuples_total_estimate} ({progress_estimate}%)"
            str_bounds = f"\nBounds: [{bounds_min}, {bounds_max}] ({bounds_progress_estimate}%)"
            str_parent = "" if "Parent Relationship" not in node else f"\nParent Relationship: {node['Parent Relationship']}"

            label = "".join([
                str_node,
                str_pipeline,
                str_tuples,
                str_bounds,
                str_parent,
            ])
            return label

        mapping = {plan_node_id: get_label(plan_node_id) for plan_node_id in self._graph.nodes}
        graph = relabel_nodes(self._graph, mapping, copy=True)
        agraph = to_agraph(graph)
        agraph.layout("dot")
        agraph.draw(filepath)
