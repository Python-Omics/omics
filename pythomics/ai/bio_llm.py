from __future__ import annotations
import os

from openai import OpenAI

# TODO: hacer que parámetros como temperature, max_tokens sean configurables desde
# inicialización?

class BioLLM:
    """
    Clase encargada de manejar comunicación con LLMs.
    Caso de uso base: intepretación y análisis preliminar
    de firma multiómica

    Se usa por defecto OpenRouter, permitiendo acceso a modelos
    gratuitos.
    """
    def __init__(self, api_key: str = None, provider: str = "openrouter", model: str = "meta-llama/llama-3-8b-instruct:free"):
        """
        Se inicializa cliente LLM

        Params:
        api_key: str
            API key para acceder al proveedor de LLMs. 
            Si None, se busca la variable de entorno 'OPENROUTER_API_KEY'
        provider: str
            Proveedor de LLMs a usar: 'openrouter', 'openai', 'groq'.
        model: str
            Nombre del modelo a usar. Por defecto, se usa modelo gratuito
            'meta-llama/llama-3-8b-instruct:free'
        """
        self.model = model
        self.provider = provider
        
        if provider == "openrouter":
            self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
            base_url = "https://openrouter.ai/api/v1"
        elif provider == "openai":
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            base_url = "https://api.openai.com/v1"
        elif provider == "groq":
            self.api_key = api_key or os.environ.get("GROQ_API_KEY")
            base_url = "https://api.groq.com/openai/v1"
        else:
            raise ValueError(f"Proveedor '{provider}' no soportadao.")

        self.client = OpenAI(
            base_url=base_url,
            api_key=self.api_key
        ) 

        
        # TODO: revisar esto si necesario. generado por gemini
        self.system_prompt = (
            "Eres un experto bioinformático y oncólogo computacional de nivel senior. "
            "Tu tarea es analizar firmas multiómicas (biomarcadores) extraídas de modelos estadísticos integrativos (ej. sPLS-DA/DIABLO). "
            "Se te proporcionará una lista de biomarcadores (genes, proteínas, miRNAs, etc.) organizados por bloque ómico, "
            "junto con su 'Loading' (peso e importancia en el modelo) y su función biológica conocida.\n\n"
            "REGLAS ESTRICTAS:\n"
            "1. REDACCIÓN: Sé conciso, profesional y clínico. No uses saludos ('Hola', 'Aquí tienes el reporte'). Ve directo al análisis.\n"
            "2. ESTRUCTURA: Organiza tu respuesta en:\n"
            "   - Resumen Global (1 párrafo con la 'fotografía general' de los procesos alterados).\n"
            "   - Hallazgos por Bloque Ómico (Qué vías destacan en mRNA, Proteínas, etc., basándote en los loadings altos).\n"
            "   - Conclusión Biológica (Potenciales implicaciones funcionales conjuntas o sugerencias para investigar).\n"
            "3. RIGOR: Basa tu análisis EXCLUSIVAMENTE en la información y las funciones proporcionadas en el contexto. No inventes funciones de genes que digan 'Sin información'. Si la mayoría dice 'Sin información', menciónalo como una limitación.\n"
            "4. SIGNIFICADO DEL LOADING: Variables con Loadings positivos vs negativos indican expresión en grupos contrastantes o direcciones opuestas en el componente latente. Analiza los contrastes si es posible."
        )

    def generate_analysis(self, context_string: str):
        """
        Se envía el contexto estructurado al LLM para generar análisis
        preliminar
        """
        if not context_string or str(context_string).strip() == "":
            return "No se proporcionó contexto de biomarcadores para analizar."
        
        print(f"Enviando datos al LLM ({self.model}) en {self.provider}...")
        
        try: 
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": context_string}
                ],
                temperature=0.25, # TODO: AJUSTAR ESTE VALOR, PERO MANTENER SIEMPRE BAJO
                max_tokens=3000
            )

            analysis_text = response.choices[0].message.content
            
            finish_reason = response.choices[0].finish_reason
            if finish_reason == 'length':
                print("Advertencia: El modelo alcanzó el límite de tokens o tiempo del servidor.")
                analysis_text += "\n\n---\n**[ADVERTENCIA DEL SISTEMA]: El análisis fue interrumpido en este punto debido a los límites de longitud (tokens) o tiempo de espera del proveedor de IA.**"

            print("Análisis LLM completado.")
            return analysis_text

        except Exception as e:
            error_msg = f"Error al generar análisis con LLM: {str(e)}"
            print(error_msg)
            return error_msg
