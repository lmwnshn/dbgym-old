# QPPNet

The implementation of QPPNet is based on the following paper, particularly the appendix in the arxiv version:

> Plan-Structured Deep Neural Network Models for Query Performance Prediction
> Ryan Marcus, Olga Papaemmanouil
> https://www.vldb.org/pvldb/vol12/p1733-marcus.pdf
> https://arxiv.org/abs/1902.00132

However, there have since been updates to PostgreSQL.
The most obvious change is that there are additional EXPLAIN features such as node types.
You can generally find all the node types from reading explain.c.
https://github.com/postgres/postgres/blob/master/src/backend/commands/explain.c

Additionally, QPPNet requires the tree structure of a query to be captured so that
the children nodes of a plan can feed input vectors into parent nodes.
However, the gym (and more generally pandas) likes to operate on tabular data.
We adopt a naive tree encoding where [0] = root; [0,0] = root,child0; [0,1] = root,child1; etc.


Feature             PostgreSQL ops  Encoding    Description
Observation Index   All             Numeric     The index of this observation.
! Children Indexes  All             [Numeric]   List of observation indexes for direct children nodes.
! Query Num         All             Numeric     The number of this query, used to associate datapoints together.
! Node Position     All             [Numeric]   Node location.
                                                [0] = root; [0,0] = root,child0; [0,1] = root,child1; etc.
! Query Hash        All             [Numeric]   Query hash, used to associate datapoints together.
! Node Type         All             One-hot     The node type, added because we flatten to a common format.
Plan Width          All             Numeric     Optimizer's estimate of the width of each output row
Plan Rows           All             Numeric     Optimizer's estimate of the cardinality of the output
                                                of the operator
Plan Buffers        All             Numeric     Optimizer's estimate of the memory requirements
                                                of an operator
Estimated I/Os      All             Numeric     Optimizer's estimate of the number of I/Os performed
Total Cost          All             Numeric     Optimizer's cost estimate for this operator, plus the subtree
Join Type           Joins           One-hot     ! One of: Inner, Left, Full, Right, Semi, Anti
Parent Relationship Joins           One-hot     When the child of a join.
                                                ! One of: Inner, Outer, Subquery, Member, children, child
Hash Buckets        Hash            Numeric     # hash buckets for hashing
Hash Algorithm      Hash            One-hot     Hashing algorithm used
Sort Key            Sort            One-hot     Key for sort operator
Sort Method         Sort            One-hot     Sorting algorithm, e.g. "quicksort", "top-N heapsort",
                                                "external sort"
                                                ! One of: still in progress, top-N heapsort, quicksort,
                                                          external sort, external merge
Relation Name       All Scans       One-hot     Base relation of the leaf
Attribute Mins      All Scans       Numeric     Vector of minimum values for relevant attributes
Attribute Medians   All Scans       Numeric     Vector of median values for relevant attributes
Attribute Maxs      All Scans       Numeric     Vector of maximum values for relevant attributes
Index Name          Index Scans     One-hot     Name of index
Scan Direction      Index Scans     ! One-hot   Direction to read the index (Backward, NoMovement, Forward)
Strategy            Aggregate       One-hot     ! One of: plain, sorted, hashed, mixed
Partial Mode        Aggregate       ! One-hot   Eligible to participate in parallel aggregation
                                                One of: Simple, Partial, Finalize
Operation           Aggregate       One-hot     The aggregation to perform, e.g. max, min, avg
