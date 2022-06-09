from collections import defaultdict
import mmh3

outgoing_properties = defaultdict(set)
incoming_properties = defaultdict(set)
# Dict[Set[str]]

with open("inputfile") as inputfile:
    for triple_string in inputfile:
        triple_list = triple_string.split(" ", maxsplit=2)
        s, p, o = triple_list[0], triple_list[1], triple_list[2]
        outgoing_properties[s].add(p)
        if o.startswith("\""):
            incoming_properties['http://example.org/literal'].add(p)
        else:
            incoming_properties[o].add(p)


# convert all sets to hashes.

outgoing_properties_hashed = {}
for (s, p) in outgoing_properties:
    property_hash = mmh3.hash128(','.join(sorted(list(p)))).encode('utf8')
    outgoing_properties_hashed[s] = property_hash


incoming_properties_hashed = {}
for (s, p) in incoming_properties:
    property_hash = mmh3.hash128(','.join(sorted(list(p)))).encode('utf8')
    incoming_properties_hashed[s] = property_hash


incoming_and_outgoing_properties_hashed = {}

for entity in set(incoming_properties.keys()).union(set(outgoing_properties.keys())):
    incoming = incoming_properties_hashed[entity] if entity in incoming_properties_hashed else 0
    outgoing = outgoing_properties_hashed[entity] if entity in outgoing_properties_hashed else 0
    combined_hash = + outgoing_properties_hashed[entity]
    incoming_and_outgoing_properties_hashed[s] = combined_hash


with open("inputfile") as inputfile:
    for triple_string in inputfile:
        # output
        # split
        triple_list = triple_string.split(" ", maxsplit=2)
        s, p, o = triple_list[0], triple_list[1], triple_list[2]
        if o.startswith("\""):
            # TODO
            pass
        else:
            print(outgoing_properties_hashed[s],
                  p, outgoing_properties_hashed[o])