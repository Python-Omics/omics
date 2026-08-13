import pandas as pd

from pythomics.ai import BioAnnotator
from pythomics.reports import BioReportPDF



if __name__ == "__main__":
    
    # carga biomarcadores df
    biomarkers_arn = pd.read_csv("results/tables/biomarcadores_mRNA_Elite_Python.csv", index_col=0)
    biomarkers_prot = pd.read_csv("results/tables/biomarcadores_Proteinas_Elite_Python.csv", index_col=0)

    annotator = BioAnnotator()
    report = BioReportPDF()
    report.generate_biomarkers_report(
        output_path="results/reporte_test.pdf",
        biomarkers_dfs={
            'mRNA': biomarkers_arn,
            'Proteinas': biomarkers_prot
        },
        annotator_instance=annotator
    )

