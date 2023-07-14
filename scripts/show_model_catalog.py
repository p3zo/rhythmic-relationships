from rhythmic_relationships.model_utils import get_model_catalog

catalog = get_model_catalog("hits_encdec")
for ix, row in catalog.iterrows():
    print(row.name, row.config["data"]["part_1"], row.config["data"]["part_2"])
