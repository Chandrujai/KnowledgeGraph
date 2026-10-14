from langchain_community.graphs import NetworkxEntityGraph

# 1. Initialize the wrapper
graph = NetworkxEntityGraph()

# Access the underlying standard networkx graph directly
nx_g = graph._graph

# 2. Same triplets we used earlier
triplets_data = [
    ("Neo4j", "DEVELOPED_BY", "Neo4j, Inc."),
    ("Emil Eifrem", "STARTED", "Neo4j"),
    ("Alice", "WORKS_AT", "Neo4j"),
    ("Alice", "WORKS_IN", "San Francisco"),
    ("Alice", "COLLABORATES_WITH", "Bob")
]

# 3. Add nodes and edges into the underlying networkx graph
for source, rel, target in triplets_data:
    nx_g.add_edge(source, target, relation=rel)

print("Successfully stored knowledge graph data into local NetworkX graph!")

# 4. Querying the local graph using LangChain's built-in helper
alice_knowledge = graph.get_entity_knowledge("Alice")
print("\nKnowledge surrounding 'Alice':")
print(alice_knowledge)