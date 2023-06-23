import mygene


def convert_gene_name(ensembl_ids, ticks=True):
    mg = mygene.MyGeneInfo()
    gene_symbol_names = []
    for id in ensembl_ids:
        if ticks:
            id = id.get_text()
        # Note: this requires an internet connection
        query = mg.query(id.split(".")[0],
                         scopes='ensembl.gene', fields='symbol',
                         species='human', returnall=True,
                         as_datafarame=True, size=1)
        if query['hits']:
            if len(query['hits'][0]['symbol']) < 3:
                query_add = mg.query(id.split(".")[0],
                                     scopes='ensembl.gene', fields='name',
                                     species='human', returnall=True,
                                     as_datafarame=True, size=1)
                gene_symbol_names.append(query['hits'][0]['symbol'] + "\n(" +
                                         query_add['hits'][0]['name'] + ")")
            else:
                gene_symbol_names.append(query['hits'][0]['symbol'])
        else:
            gene_symbol_names.append(id)
    return gene_symbol_names
