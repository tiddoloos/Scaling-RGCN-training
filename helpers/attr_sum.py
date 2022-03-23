from rdflib import Graph


def make_graph_from_nt(input_data):
    g = Graph()
    data = open(input_data, "rb")
    g.parse(data, format="nt")
    return g

def make_attr_sum(graph):
    # was there in allessndro's SPARQL paper
    # PREFIX rdf: <http ://www.w3. org/1999/02/22=rdf=syntax=ns#>
    # found PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> in paper pf Campinas

    attr_query = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    CONSTRUCT {
        ?hs ?p ?ho .
    }
    WHERE
    {
        {
            SELECT ?s (SHA1(group_concat(?p; separator = ",")) as ?sID) WHERE {
                SELECT DISTINCT ?s ?p {
                    ?s ?p ?_
                } order by ?p
            } group by ?s
        }
        BIND(URI(CONCAT("http://example.org/", ?sID)) AS ?hs)
        ?s ?p ?o
        OPTIONAL { 
            {
                SELECT ?o (SHA1(group_concat(?p; separator = ",")) as ?oID) WHERE { 
                    SELECT DISTINCT ?o ?p {
                        ?o ?p ?_
                    } order by ?p
                } group by ?o
            }
            BIND(URI(CONCAT("http://example.org/", ?oID)) AS ?ho2)
        }
        BIND(IF(BOUND(?oID), ?ho2, ?o) AS ?ho)
    }
    """
    results = graph.query(attr_query)
    print(results)
    # return [str(result[0]) for result in results]

def main_attr_sum(filename):
    graph = make_graph_from_nt(filename)
    print(len(graph))
    sum_graph = make_attr_sum(graph)

filename = "../../data/AIFB/raw/aifb_stripped.nt"
main_attr_sum(filename)
