import os
import pandas as pd
from typing import Dict, Optional, Union, List
from fpdf import FPDF
from matplotlib import font_manager
import re
import textwrap
import logging

logger = logging.getLogger(__name__)

class BioReportPDF(FPDF):
    """
    Class for generating PDF reports.
    """
    def __init__(self, title = "Report Multiómico", font_path: Optional[str] = None, font_family: str = "DejaVu Sans") :
        super().__init__()
        self.report_title = title
        self.set_auto_page_break(auto=True, margin=15)
        self._setup_font(font_path, font_family)

    def _setup_font(self, font_path: Optional[str] = None, font_family: str = "DejaVu Sans"):
        """
        Registers a portable Unicode font if available; otherwise, uses Helvetica.

        Resolution:
        1. If `font_path` is passed, that TTF is used directly.
        2. Otherwise, the `font_family` is looked up with matplotlib.font_manager
           (which includes its own copy of DejaVu Sans, portable to any OS).
        3. Last resort: Helvetica (Latin-1).
        """
        family = "Helvetica"
        if font_path and os.path.exists(font_path):
            self._register_font(font_path)
            family = "Unicode"
        else:
            try:
                fp_regular = font_manager.findfont(
                    font_manager.FontProperties(family=font_family, weight="normal"))
                try:
                    fp_bold = font_manager.findfont(
                        font_manager.FontProperties(family=font_family, weight="bold"))
                except Exception:
                    fp_bold = fp_regular
                try:
                    fp_italic = font_manager.findfont(
                        font_manager.FontProperties(family=font_family, style="italic"))
                except Exception:
                    fp_italic = fp_regular
                self._register_font(fp_regular, fp_bold, fp_italic)
                family = "Unicode"
            except Exception:
                family = "Helvetica"
        self._font_family = family
        self.set_font(family, size=11)

    def _register_font(self, regular: str, bold: Optional[str] = None, italic: Optional[str] = None):
        """Registers a 'Unicode' family with regular/bold/italic styles."""
        self.add_font("Unicode", "", regular)
        self.add_font("Unicode", "B", bold or regular)
        self.add_font("Unicode", "I", italic or regular)

    def header(self):
        """
        Header for each page
        """
        self.set_font(self._font_family, 'B', 16)
        self.set_text_color(44, 62, 80) # Dark blue
        self.cell(0, 10, self.report_title, border=0, align='C', new_x="LMARGIN", new_y="NEXT")
        self.line(10, 20, 200, 20) # Separator line
        self.ln(5)
        
    def footer(self):
        """
        Page footer
        """
        self.set_y(-15)
        self.set_font(self._font_family, 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Página {self.page_no()}', align='C')
    
    def _add_section_title(self, title: str):
        """Adds a section title with a colored background"""
        self.ln(5)
        self.set_font(self._font_family, 'B', 12)
        self.set_fill_color(236, 240, 241) # light gray for title background
        self.set_text_color(44, 62, 80)
        self.cell(0, 10, f"  {title}", border=0, fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(4)
    
    def _add_plot(self, image_path: Optional[str], width: int = 160):
        """Inserts a centered image in the PDF. If the path is None, it draws nothing."""
        if not image_path:
            return
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            self.set_font(self._font_family, 'B', 10)
            self.set_text_color(231, 76, 60) # error red
            self.cell(0, 10, f"[Error: Gráfico no encontrado en {image_path}]", new_x="LMARGIN", new_y="NEXT")
            self.set_text_color(0, 0, 0)
            return
        
        x_pos = (210 - width) / 2
        if self.get_y() > 220: # if too far down, jump to a new page
            self.add_page()
            
        self.image(image_path, x=x_pos, w=width)
        self.ln(5)
    
    def _add_multiple_plots(self, image_paths: List[str], width: int = 160):
        """Inserts multiple centered images in the PDF. If the list is empty, it draws nothing."""
        for path in image_paths:
            self._add_plot(path, width=width)
            self.ln(10)

    def _add_biomarker_table(self, block_name: str, df: pd.DataFrame, annotations: Dict[str, dict], top_n: int = 15):
        """Draws a table with the biomarkers and their functional translation"""
        self.set_font(self._font_family, 'B', 11)
        self.set_text_color(0, 0, 0)
        self.cell(0, 8, f"Bloque: {block_name} (Top {top_n})", new_x="LMARGIN", new_y="NEXT")
        
        col_widths = [35, 40, 20, 95]
        headers = ["ID Original", "Símbolo", "Loading", "Descripción"]
        
        # draw headers
        self.set_font(self._font_family, 'B', 9)
        self.set_fill_color(189, 195, 199)
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 8, header, border=1, fill=True, align='C')
        self.ln()
        
        # draw rows
        self.set_font(self._font_family, '', 8)
        df_top = df.head(top_n)
 
        line_height = 5

        for idx, row in df_top.iterrows():
            loading = row.get('value', 0.0)
            info = annotations.get(str(idx), {})
            
            symbol = str(info.get('official_symbol', 'N/A'))[:20]
            summary_full = str(info.get('summary', 'Sin información.'))
            
            # get the width of the description column (95)
            desc_width = col_widths[3]
            lines = self.multi_cell(desc_width, line_height, summary_full, split_only=True)
            num_lines = len(lines)
            row_height = line_height * num_lines
            
            # check for a page break if the row does not fit
            if self.get_y() + row_height > self.page_break_trigger:
                self.add_page()
            
            # initial position
            x_start = self.get_x()
            y_start = self.get_y()

            self.cell(col_widths[0], row_height, str(idx)[:20], border=1)
            self.cell(col_widths[1], row_height, symbol, border=1)

            # colors for the Loading value
            if loading < 0:
                self.set_text_color(192, 57, 43) # red if negative
            else:
                self.set_text_color(39, 174, 96) # green if positive
                
            self.cell(col_widths[2], row_height, f"{loading:.4f}", border=1, align='C')
            self.set_text_color(0, 0, 0) # black
            
            x_desc = self.get_x()
            self.multi_cell(desc_width, line_height, summary_full, border=1, new_x="LMARGIN", new_y="NEXT")

            self.set_xy(x_start, y_start + row_height)
            
        self.ln(5)

    def generate_biomarkers_report(
        self,
        output_path: str,
        biomarkers_dfs: Dict[str, pd.DataFrame],
        annotator_instance,
        biomarkers_plot_path: Optional[Union[str, List[str]]] = None,
        report_title: Optional[str] = None,
        llm_instance = None # optional param. if no llm is passed, no llm analysis is done
    ):
        """
        Generates a PDF report.
        Simply create a BioReportPDF and call this method.

        Params:
        output_path: str
            path where the PDF will be saved (e.g., 'results/report.pdf')
        biomarkers_dfs: dict[str, pd.DataFrame]
            dictionary with biomarker DataFrames per block
        annotator_instance: pythomics.ai.BioAnnotator
            BioAnnotator instance to translate IDs and obtain biomarker
            information
        biomarkers_plot_path: str/list[str]
            single path or list of paths to images to add to the report
        report_title: str
            main title of the document
        llm_instance: pythomics.ai.BioLLM
            previously initialized BioLLM instance. If provided,
            the report will generate a preliminary biomarker analysis using LLM.
        """
        if report_title:
            self.report_title = report_title
        logger.info(f"Generating PDF report: '{self.report_title}'...")
        self.add_page()
        
        # -- plots --
        if biomarkers_plot_path:
            self._add_section_title("1. Firma Molecular Seleccionada")
            
            # handle whether a single string or a list of plot paths was passed
            if isinstance(biomarkers_plot_path, str):
                self._add_plot(biomarkers_plot_path, width=190)
            elif biomarkers_plot_path:
                self._add_multiple_plots(biomarkers_plot_path, width=170)

        # -- biomarker tables and information --
        self.add_page()
        self._add_section_title("2. Anotación Funcional de Biomarcadores")
        
        gene_centric = ['mRNA', 'CNV', 'Methy', 'Transcriptomics', 'Proteomics']
        
        for block, df in biomarkers_dfs.items():
            feature_ids = df.index.tolist()
            
            # On-the-fly annotation routing
            if 'miRNA' in block or 'microRNA' in block:
                annotations = annotator_instance.annotate_mirnas(feature_ids)
            elif 'Prot' in block: # Covers Proteomics, Proteins, etc
                annotations = annotator_instance.annotate_proteins(feature_ids)
            elif any(g in block for g in gene_centric):
                annotations = annotator_instance.annotate_genes(feature_ids)
            elif 'Metabol' in block:
                annotations = annotator_instance.annotate_metabolites(feature_ids)
            else:
                annotations = annotator_instance.annotate_genes(feature_ids)
            self._add_biomarker_table(block_name=block, df=df, annotations=annotations, top_n=15)

        # -- LLM analysis --
        self.add_page()
        self._add_section_title("3. Análisis y Conclusiones Biológicas Preliminares (Generado por LLM)")
        self.set_font(self._font_family, '', 10)
        
        # TODO: add which LLM model was used?
        
        disclaimer_text = (
            "La totalidad del análisis presentado a continuación es generado automáticamente mediante "
            "Inteligencia Artificial (LLM). Su propósito es ofrecer un análisis estrictamente exploratorio "
            "y orientativo preliminar. Por lo tanto, no debe, bajo ninguna circunstancia, considerarse como "
            "una conclusión biológica o clínica definitiva."
        )


        if llm_instance:
            # get the context string using the annotator instance
            context_string = annotator_instance.get_llm_context_string(biomarkers_dfs)
            
            # generate LLM analysis
            llm_analysis_text = llm_instance.generate_analysis(context_string)

            # AI warning box
            self.set_fill_color(253, 245, 230)
            self.set_draw_color(211, 84, 0)
            self.set_text_color(160, 64, 0)
            self.set_font(self._font_family, 'B', 9)
            self.multi_cell(0, 6, disclaimer_text, border=1, fill=True, align='C')
            self.ln(5) 
            
            self.set_draw_color(0, 0, 0)
            self.set_text_color(0, 0, 0)
            
            # force a blank space every 80 characters to avoid problems
            # of the llm generating very long sequences without spaces that break everything
            safe_text = re.sub(r'([^\s]{80})', r'\1 ', str(llm_analysis_text))
            
            # -- "parse" the text to simulate markdown --
            lines = safe_text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    self.ln(4)
                    continue
                
                # detect dividers
                if line == '---':
                    self.ln(2)
                    self.line(self.get_x(), self.get_y(), 195, self.get_y())
                    self.ln(4)
                    continue

                base_size = 10
                style_normal = ''
                style_bold = 'B'
                style_italic = 'I'

                # detect headers (###, ####...)
                header_match = re.match(r'^(#+)\s+(.*)', line)
                if header_match:
                    level = len(header_match.group(1))
                    line = header_match.group(2)

                    # adjust font size according to header level
                    base_size = 14 if level <= 2 else (12 if level == 3 else 11)
                    self.set_font(self._font_family, 'B', base_size)
                    
                    style_normal = 'B'
                    style_italic = 'I'
                    self.ln(4)

                # detect if line is a bullet point
                if line.startswith('* ') or line.startswith('- '):
                    line = line[2:] # remove the leading asterisk/hyphen
                    self.set_x(self.l_margin + 5)
                    self.write(6, "• ")
                
                # regex to separate normal text from bold text (between **)
                parts = re.split(r'\*\*(.*?)\*\*', line)

                for i, part in enumerate(parts):
                    if not part:
                        continue
                    
                    if i % 2 == 1:
                        # it is bold
                        self.set_font(self._font_family, style_bold, base_size)
                        try:
                            self.write(6, part)
                        except Exception:
                            broken_part = " ".join([part[j:j+60] for j in range(0, len(part), 60)])
                            self.write(6, broken_part)
                    else:
                        # regex to separate italic text (between *)
                        sub_parts = re.split(r'\*(.*?)\*', part)
                        
                        for k, sub_part in enumerate(sub_parts):
                            if not sub_part:
                                continue

                            if k % 2 == 1:
                                # it is italic
                                self.set_font(self._font_family, style_italic, base_size)
                            else:
                                # it is normal text
                                self.set_font(self._font_family, style_normal, base_size)

                            try:
                                self.write(6, sub_part)
                            except Exception:
                                broken_part = " ".join([sub_part[m:m+60] for m in range(0, len(sub_part), 60)])
                                self.write(6, broken_part)

                # end line
                self.ln(6)
                self.set_x(self.l_margin) # reset X position to left margin

        else:
            # warning box
            self.set_fill_color(245, 245, 245)
            self.set_draw_color(200, 200, 200)
            self.set_text_color(128, 128, 128)
            self.set_font(self._font_family, 'I', 9)

            self.multi_cell(0, 6, disclaimer_text, border=1, fill=True, align='C')

        # -- save report --
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.output(output_path)
        logger.info(f"PDF report saved at: {output_path}")
