import pandas as pd
from biothings_client import get_client
import requests
import logging

logger = logging.getLogger(__name__)

class BioAnnotator:
    """
    Class in charge of querying biological APIs to translate biomarker IDs into
    human-readable names and to retrieve relevant information.
    """
    def __init__(self):
        # instantiate the specific clients for each biomarker type
        self.gene_client = get_client('gene')
        self.chem_client = get_client('chem')
        self.cache = {}

    def annotate_genes(self, gene_list, species='all'):
        """
        Takes a list of gene IDs and returns a dictionary with the
        full name and functional summary (if available). 
        MyGene is used for the query (works for mRNA, CNV, Proteins and Methylation mapped to genes).
        
        Params:
        gene_list: list
            List of gene IDs to query
        species: str
            Species of the genes (default 'all')
        
        Returns:
        dict
            Dictionary with information for each gene, including:
            - official_symbol: official gene symbol
            - name: full gene name
            - summary: functional summary of the gene
        """
        logger.info(f"Querying identifiers for {len(gene_list)} genes/CNVs...")

        # -- normalization (handling variables coming from R) --
        clean_to_original = {str(gene).replace('.', '-'): gene for gene in gene_list}
        clean_genes = list(clean_to_original.keys())

        # -- cache: if this query was already made, return what was stored --
        cache_key = ('genes', species, frozenset(clean_genes))
        if cache_key in self.cache:
            return self.cache[cache_key]

        # -- MyGene.info query --
        annotations = {}
        try:
            results = self.gene_client.querymany(
                clean_genes,
                scopes='symbol,entrezgene,ensemblgene,alias,reporter,uniprot',
                fields='name,symbol,summary',
                species=species,
                returnall=False,
                verbose=False
            )

            # -- result processing --
            for result in results:
                clean_query = result.get('query')
                original_query = clean_to_original.get(clean_query, clean_query)

                if original_query not in annotations:
                    name = result.get('name', 'Name not found')
                    summary = result.get('summary', '')
                
                    if not summary or str(summary).strip() == '':
                        summary = f"Gene/Protein: {name}"

                    annotations[original_query] = {
                        'official_symbol': result.get('symbol', original_query),
                        'name': name,
                        'summary': summary
                    }
        except Exception as e:
            logger.exception(f"Error querying MyGene.info: {e}")

        # -- handle IDs not found --
        for clean_gene, orig_gene in clean_to_original.items():
            if orig_gene not in annotations:
                annotations[orig_gene] = {
                    'official_symbol': orig_gene,
                    'name': 'Unknown or unmapped ID',
                    'summary': 'No information in databases'
                }
        
        self.cache[cache_key] = annotations
        return annotations
    
    def annotate_proteins(self, protein_list):
        """
        Takes a list of UniProt IDs and uses the official UniProt REST API
        to get the protein name and its biological function.
        """
        logger.info(f"Querying UniProt REST API for {len(protein_list)} proteins...")
        annotations = {}
        
        # clean protein IDs
        clean_to_original = {str(protein).split('-')[0]: protein for protein in protein_list}
        clean_proteins = list(clean_to_original.keys())

        # -- cache: if this query was already made, return what was stored --
        cache_key = ('proteins', frozenset(clean_proteins))
        if cache_key in self.cache:
            return self.cache[cache_key]

        # queries are done in batches to avoid problems
        batch_size = 5
        
        for i in range(0, len(clean_proteins), batch_size):
            batch = clean_proteins[i:i + batch_size]
            
            # build the query
            query_str = " OR ".join([f"accession:{acc}" for acc in batch])
            
            # build the url for the UniProt REST API query
            url = f"https://rest.uniprot.org/uniprotkb/search?query=({query_str})&fields=accession,protein_name,cc_function&format=json"
            try:
                response = requests.get(url, timeout=10) # NOTE: add timeout if issues

                if response.status_code == 200: # if 200 OK, process the JSON response
                    results = response.json().get('results', [])

                    for result in results:
                        try:
                            accession = result.get('primaryAccession')
                            original_query = clean_to_original.get(accession, accession)
                            
                            # extract name
                            name_obj = result.get('proteinDescription', {})
                            name = "Name not available"
                        
                            if 'recommendedName' in name_obj:
                                name = name_obj['recommendedName']['fullName']['value']
                            elif 'submissionNames' in name_obj and len(name_obj['submissionNames']) > 0:
                                name = name_obj['submissionNames'][0]['fullName']['value']
                            elif 'alternativeNames' in name_obj and len(name_obj['alternativeNames']) > 0:
                                name = name_obj['alternativeNames'][0]['fullName']['value']
                            
                            # extract biological function
                            summary = ""
                            comments = result.get('comments', [])
                            for comment in comments:
                                if comment.get('commentType') == 'FUNCTION':
                                    texts = comment.get('texts', [])
                                    if texts:
                                        summary = texts[0].get('value', '')
                                        break
                            
                            if not summary or summary.strip() == '':
                                summary = f"Name/Family: {name}"

                            annotations[accession] = {
                                'official_symbol': accession,
                                'name': name,
                                'summary': summary
                            }
                        except Exception as inner_er:
                            logger.exception(f"Error parsing UniProt record {accession}: {inner_er}")
                            continue # move on to the next record if it fails
                else:
                    logger.warning(f"Error querying UniProt batch, HTTP status: {response.status_code}")
                    
            except Exception as e:
                logger.exception(f"Error querying UniProt batch: {e}")

        # -- handle IDs not found --
        for clean_protein, orig_protein in clean_to_original.items():
            if orig_protein not in annotations:
                annotations[orig_protein] = {
                    'official_symbol': orig_protein,
                    'name': 'Unknown or unmapped ID',
                    'summary': 'No information in UniProt'
                }

        self.cache[cache_key] = annotations
        return annotations

    # TODO: no API query is made here for now
    def annotate_mirnas(self, mirna_list, species='human'):
        """
        Takes a list of miRNA IDs and translates them.
        
        """
        logger.info(f"Querying identifiers for {len(mirna_list)} miRNAs...")
        
        # -- id normalization --
        clean_to_original = {mir.replace('.', '-'): mir for mir in mirna_list}
        annotations = {}

        # -- cache: if this query was already made, return what was stored --
        cache_key = ('mirnas', species, frozenset(clean_to_original))
        if cache_key in self.cache:
            return self.cache[cache_key]

        # NOTE: No estoy seguro de esto. Se supone que es una 'regla universal'
        # para todos los miRNAs. Su único propósito sería dar más contexto a los LLMs.
        mirna_context = (
            "MicroRNA (ncRNA). Regulates gene expression post-transcriptionally by "
            "binding to the 3'-UTR of target mRNAs, inducing their degradation or "
            "translational repression. In cancer, they can act as oncomiRs or tumor suppressors."
        )
        
        for clean_mirna, orig_mirna in clean_to_original.items():
            annotations[orig_mirna] = {
                'official_symbol': clean_mirna,
                'name': f'MicroRNA {clean_mirna}',
                'summary': mirna_context,
                '_resolved': False # in case we later want to fetch from Ensembl
            }

        self.cache[cache_key] = annotations
        return annotations

    def annotate_metabolites(self, metabolite_list):
        """
        Takes a list of metabolite IDs and returns a dictionary with
        the full name and functional summary (if available).
        PubChem PUG REST is used for the query.
        """
        logger.info(f"Querying identifiers for {len(metabolite_list)} metabolites...")
        annotations = {}

        # -- cache: if this query was already made, return what was stored --
        cache_key = ('metabolites', frozenset(metabolite_list))
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        for metabolite in metabolite_list:
            # build the query url
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{metabolite}/property/IUPACName,MolecularFormula,MolecularWeight/JSON"
            
            try:
                response = requests.get(url, timeout=15)
                if response.status_code == 200:
                    props = response.json().get('PropertyTable', {}).get('Properties', [])
                    if props:
                        best = props[0]
                        name = best.get('IUPACName', metabolite)
                        summary = (
                            f"Molecular formula: {best.get('MolecularFormula', 'N/A')}; "
                            f"Weight: {best.get('MolecularWeight', 'N/A')}; "
                            f"IUPAC: {name}"
                        )
                        annotations[metabolite] = {
                            'official_symbol': name,
                            'name': name,
                            'summary': summary
                        }
                        continue
                else:
                    logger.warning(f"Error querying PubChem for {metabolite}, HTTP status: {response.status_code}")

            except Exception as e:
                logger.warning(f"Error querying PubChem for {metabolite}: {e}")

            # synonyms (to at least get a readable name)
            try:
                syn_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{metabolite}/synonyms/JSON"
                response = requests.get(syn_url, timeout=15)
                if response.status_code == 200:
                    info_list = response.json().get('InformationList', {}).get('Information', [])
                    if info_list:
                        syn = info_list[0].get('Synonym', [])
                        if syn:
                            annotations[metabolite] = {
                                'official_symbol': syn[0],
                                'name': syn[0],
                                'summary': 'No functional description in PubChem (only synonyms).'
                            }
                            continue
            except Exception as e:
                logger.warning(f"Error querying synonyms in PubChem for {metabolite}: {e}")

            annotations[metabolite] = {
                'official_symbol': metabolite,
                'name': 'Unknown or unmapped ID',
                'summary': 'No information in PubChem'
            }

        self.cache[cache_key] = annotations
        return annotations

    def get_llm_context_string(self, biomarkers_dict):
        """
        Generates a structured string that can be passed to an LLM.
        
        Params:
        biomarkers_dict: dict
            Dictionary with omic blocks and their respective biomarker DataFrames.

        Returns:
        str
            String with biomarker information, to be used as context in LLMs.

        """
        context_lines = ["### Extracted biomarker context ###\n"]
        
        gene_centric_blocks = ['mRNA', 'CNV', 'Methy', 'Proteomics', 'Transcriptomics']

        for block_name, df in biomarkers_dict.items():
            context_lines.append(f"-- Omic block {block_name} --")
            feature_ids = df.index.tolist()
            
            # route to the corresponding function based on block type
            if 'miRNA' in block_name or 'microRNA' in block_name:
                annotations = self.annotate_mirnas(feature_ids)
            elif 'Prot' in block_name:
                annotations = self.annotate_proteins(feature_ids)
            elif any(name in block_name for name in gene_centric_blocks):
                annotations = self.annotate_genes(feature_ids)
            elif 'Metabol' in block_name:
                annotations = self.annotate_metabolites(feature_ids)
            else:
                annotations = self.annotate_genes(feature_ids)

            for idx, row in df.iterrows():
                loading = row.get('value', 0.0) 
                info = annotations.get(idx, {})
                
                line = f"- ID: {idx} | Loading: {loading:.4f} | Name: {info.get('name')} | Function: {info.get('summary')}"
                context_lines.append(line)

            context_lines.append("\n") # separator between blocks

        return "\n".join(context_lines) # return the full string for the LLM


