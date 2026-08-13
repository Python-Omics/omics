import pandas as pd
from biothings_client import get_client
import requests

class BioAnnotator:
    """
    Clase encargada de consultar APIs biológicas para traducir IDs de biomarcadores a
    nombres legibles y extraer información relevante.
    """
    def __init__(self):
        # se instancian los cliente específicos para cada tipo de biomarcador
        self.gene_client = get_client('gene')
        self.chem_client = get_client('chem')
        self.cache = {}

    def annotate_genes(self, gene_list, species='all'):
        """
        Toma una lista de IDs de genes y devuelve un diccionario con el
        nombre completo y resumen funcional (si existe). 
        Se usa MyGene para la consulta (sirve para mRNA, CNV, Proteínas y Metilación mapeada a genes).
        
        Params:
        gene_list: list
            Lista de IDs de genes a consultar
        species: str
            Especie de los genes (por defecto 'all')
        
        Returns:
        dict
            Diccionario con la información de cada gen, incluyendo:
            - official_symbol: símbolo oficial del gen
            - name: nombre completo del gen
            - summary: resumen funcional del gen
        """
        print(f"Consultando identificadores de {len(gene_list)} genes/CNVs...")

        # -- normalización (manejo de variables provenientes de R) --
        clean_to_original = {str(gene).replace('.', '-'): gene for gene in gene_list}
        clean_genes = list(clean_to_original.keys())

        # -- cache: si esta consulta ya se hizo, devolver lo guardado --
        cache_key = ('genes', species, frozenset(clean_genes))
        if cache_key in self.cache:
            return self.cache[cache_key]

        # -- consulta a MyGene.info --
        results = self.gene_client.querymany(
            clean_genes,
            scopes='symbol,entrezgene,ensemblgene,alias,reporter,uniprot',
            fields='name,symbol,summary',
            species=species,
            returnall=False,
            verbose=False
        )

        # -- procesamiento de resultados --
        annotations = {}
        for result in results:
            clean_query = result.get('query')
            original_query = clean_to_original.get(clean_query, clean_query)

            if original_query not in annotations:
                name = result.get('name', 'Nombre no encontrado')
                summary = result.get('summary', '')
                
                if not summary or str(summary).strip() == '':
                    summary = f"Gen/Proteína: {name}"

                annotations[original_query] = {
                    'official_symbol': result.get('symbol', original_query),
                    'name': name,
                    'summary': summary
                }

        # -- manejo IDs no encontrados --
        for clean_gene, orig_gene in clean_to_original.items():
            if orig_gene not in annotations:
                annotations[orig_gene] = {
                    'official_symbol': orig_gene,
                    'name': 'ID desconocido o no mapeado',
                    'summary': 'Sin información en bases de datos'
                }
        
        self.cache[cache_key] = annotations
        return annotations
    
    def annotate_proteins(self, protein_list):
        """
        Toma una lista de IDs de UniProt y usa la API REST oficial de UniProt
        para obtener el nombre de la proteína y su función biológica.
        """
        print(f"Consultando UniProt REST API para {len(protein_list)} proteínas...")
        annotations = {}
        
        # limpieza ids de proteinas
        clean_to_original = {str(protein).split('-')[0]: protein for protein in protein_list}
        clean_proteins = list(clean_to_original.keys())

        # -- cache: si esta consulta ya se hizo, devolver lo guardado --
        cache_key = ('proteins', frozenset(clean_proteins))
        if cache_key in self.cache:
            return self.cache[cache_key]

        # se hacen consultas en lotes para evitar problemas
        batch_size = 5
        
        for i in range(0, len(clean_proteins), batch_size):
            batch = clean_proteins[i:i + batch_size]
            
            # se construye la query
            query_str = " OR ".join([f"accession:{acc}" for acc in batch])
            
            # se arma la url para hacer la consulta a UniProt REST API
            url = f"https://rest.uniprot.org/uniprotkb/search?query=({query_str})&fields=accession,protein_name,cc_function&format=json"
            try:
                response = requests.get(url, timeout=10) # NOTE: agregar timeout si problemas

                if response.status_code == 200: # si 200 OK, procesar JSON respuesta
                    results = response.json().get('results', [])

                    for result in results:
                        try:
                            accession = result.get('primaryAccession')
                            original_query = clean_to_original.get(accession, accession)
                            
                            # extraer nombre
                            name_obj = result.get('proteinDescription', {})
                            name = "Nombre no disponible"
                        
                            if 'recommendedName' in name_obj:
                                name = name_obj['recommendedName']['fullName']['value']
                            elif 'submissionNames' in name_obj and len(name_obj['submissionNames']) > 0:
                                name = name_obj['submissionNames'][0]['fullName']['value']
                            elif 'alternativeNames' in name_obj and len(name_obj['alternativeNames']) > 0:
                                name = name_obj['alternativeNames'][0]['fullName']['value']
                            
                            # extraer función biológica
                            summary = ""
                            comments = result.get('comments', [])
                            for comment in comments:
                                if comment.get('commentType') == 'FUNCTION':
                                    texts = comment.get('texts', [])
                                    if texts:
                                        summary = texts[0].get('value', '')
                                        break
                            
                            if not summary or summary.strip() == '':
                                summary = f"Nombre/Familia: {name}"

                            annotations[accession] = {
                                'official_symbol': accession,
                                'name': name,
                                'summary': summary
                            }
                        except Exception as inner_er:
                            print(f"Error parseando registro UniProt {accession}: {inner_er}")
                            continue # se sigue con siguiente registro si falla
                else:
                    print(f"Error consultando lote en UniProt, HTTP status: {response.status_code}")
                    
            except Exception as e:
                print(f"Error consultando lote en Uniprot: {e}")

        # -- manejos IDs no encontrados --
        for clean_protein, orig_protein in clean_to_original.items():
            if orig_protein not in annotations:
                annotations[orig_protein] = {
                    'official_symbol': orig_protein,
                    'name': 'ID desconocido o no mapeado',
                    'summary': 'Sin información en UniProt'
                }

        self.cache[cache_key] = annotations
        return annotations

    # TODO: aqui no se hace una consulta a ninguna api por ahora
    def annotate_mirnas(self, mirna_list, species='human'):
        """
        Toma una lista de IDs de miRNAs y los traduce.
        
        """
        print(f"Consultando identificadores de {len(mirna_list)} miRNAs...")
        
        # -- normalización ids --
        clean_to_original = {mir.replace('.', '-'): mir for mir in mirna_list}
        annotations = {}

        # -- cache: si esta consulta ya se hizo, devolver lo guardado --
        cache_key = ('mirnas', species, frozenset(clean_to_original))
        if cache_key in self.cache:
            return self.cache[cache_key]

        # NOTE: aqui no estoy seguro de esto. se supone que es una 'regla universal' 
        # para todos los miRNAs. único propósito sería para darle más contexto a LLMs
        mirna_context = (
            "MicroRNA (ncRNA). Regula la expresión génica post-transcripcionalmente "
            "uniéndose al 3'-UTR de los mRNAs diana, induciendo su degradación o "
            "represión traduccional. En cáncer, pueden actuar como oncomiRs o supresores tumorales."
        )
        
        for clean_mirna, orig_mirna in clean_to_original.items():
            annotations[orig_mirna] = {
                'official_symbol': clean_mirna,
                'name': f'MicroRNA {clean_mirna}',
                'summary': mirna_context,
                '_resolved': False # por si luego queremos hacer un fetch a Ensembl
            }

        self.cache[cache_key] = annotations
        return annotations

    def annotate_metabolites(self, metabolite_list):
        """
        Toma una lista de IDs de metabolitos y devuelve un diccionario con
        el nombre completo y resumen funcional (si existe).
        Se usa PubChem PUG REST  para la consulta.
        """
        print(f"Consultando identificadores de {len(metabolite_list)} metabolitos...")
        annotations = {}

        # -- cache: si esta consulta ya se hizo, devolver lo guardado --
        cache_key = ('metabolites', frozenset(metabolite_list))
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        for metabolite in metabolite_list:
            # se arma url para consultar
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{metabolite}/property/IUPACName,MolecularFormula,MolecularWeight/JSON"
            
            try:
                response = requests.get(url, timeout=15)
                if response.status_code == 200:
                    props = response.json().get('PropertyTable', {}).get('Properties', [])
                    if props:
                        best = props[0]
                        name = best.get('IUPACName', metabolite)
                        summary = (
                            f"Fórmula molecular: {best.get('MolecularFormula', 'N/A')}; "
                            f"Peso: {best.get('MolecularWeight', 'N/A')}; "
                            f"IUPAC: {name}"
                        )
                        annotations[metabolite] = {
                            'official_symbol': name,
                            'name': name,
                            'summary': summary
                        }
                        continue

            except Exception as e:
                pass

            # sinonimos (para al menos obtener un nombre legible)
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
                                'summary': 'Sin descripción funcional en PubChem (solo sinónimos).'
                            }
                            continue
            except Exception as e:
                pass

            annotations[metabolite] = {
                'official_symbol': metabolite,
                'name': 'ID desconocido o no mapeado',
                'summary': 'Sin información en PubChem'
            }

        self.cache[cache_key] = annotations
        return annotations

    def get_llm_context_string(self, biomarkers_dict):
        """
        Genera string estructurado que se puede pasar a LLM.
        
        Params:
        biomarkers_dict: dict
            Diccionaro con bloques ómicos y sus respectivos DataFrames de biomarcadores.

        Returns:
        str
            String con información de biomarcadores, para ser usado como contexto en LLMs.

        """
        context_lines = ["### Contexto de biomarcadores extraídos ###\n"]
        
        gene_centric_blocks = ['mRNA', 'CNV', 'Methy', 'Proteomics', 'Transcriptomics']

        for block_name, df in biomarkers_dict.items():
            context_lines.append(f"-- Bloque ómico {block_name} --")
            feature_ids = df.index.tolist()
            
            # enrutamiento a función correspondiente según tipo de bloque
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
                
                # TODO: considerar truncar aquí para no saturar llm

                line = f"- ID: {idx} | Loading: {loading:.4f} | Name: {info.get('name')} | Función: {info.get('summary')}"
                context_lines.append(line)

            context_lines.append("\n") # separador entre bloques

        return "\n".join(context_lines) # se devuelve el string completo para LLM


