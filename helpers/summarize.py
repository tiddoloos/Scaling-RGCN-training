import rdflib
import rdfextras
# rdfextras.registerplugins() # so we can Graph.query()

filename = "data/AIFB/raw/aifb_stripped.nt" #replace with something interesting
# uri = "uri_of_interest" #replace with something interesting


g=rdflib.Graph()
g.parse(filename)
results = g.query("""
PREFIX rdf : <http ://www.w3. org/1999/02/22=rdf=syntax=ns#>
CONSTRUCT {
    ?hs ?p ?ho .
}
WHERE
{
    {
        SELECT ?s (SHA1(group concat(?p; separator = ”,”)) as ?sID) WHERE {
            SELECT DISTINCT ?s ?p {
                ?s ?p ? _
                } order by ?p
            } group by ?s
        }
        BIND(URI(CONCAT(”http://example.org/”, ?sID)) AS ?hs)
        ?s ?p ?o
        OPTIONAL { 
            {
                SELECT ?o (SHA1(group concat(?p; separator = ”,”)) as ?oID) WHERE { 
                    SELECT DISTINCT ?o ?p {
                        ?o ?p ? _
                    } order by ?p
                } group by ?o
            }
            BIND(URI(CONCAT(”http://example.org/”, ?oID)) AS ?ho2)
        }
        BIND(IF(BOUND(?oID), ?ho2, ?o) AS ?ho)
""")

print(results)