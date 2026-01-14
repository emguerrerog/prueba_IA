import os
import sys
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import json
from datetime import datetime
from colorama import init, Fore, Style

# Inicializar colorama
init(autoreset=True)

class ChromaDBAgent:
    def __init__(self, 
                 persist_directory: str = "./chroma_db",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 llm_provider: str = "openai",  # "openai", "ollama", "local"
                 api_key: Optional[str] = None):
        """
        Agente de IA integrado con ChromaDB
        
        Args:
            persist_directory: Directorio de ChromaDB
            embedding_model: Modelo para embeddings
            llm_provider: Proveedor del LLM
            api_key: API key para servicios externos
        """
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.llm_provider = llm_provider
        self.api_key = api_key
        
        # Inicializar componentes
        self._init_chroma()
        self._init_embeddings()
        self._init_llm()
        
        # Historial de conversación
        self.conversation_history = []
        
        print(Fore.GREEN + "🤖 Agente de IA con ChromaDB inicializado" + Style.RESET_ALL)
        print(Fore.CYAN + f"📁 Base de datos: {persist_directory}")
        print(Fore.CYAN + f"🧠 Modelo: {embedding_model}")
        print(Fore.CYAN + f"💭 LLM: {llm_provider}")
    
    def _init_chroma(self):
        """Inicializar cliente de ChromaDB"""
        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            print(Fore.GREEN + "✅ ChromaDB inicializado" + Style.RESET_ALL)
        except Exception as e:
            print(Fore.RED + f"❌ Error inicializando ChromaDB: {e}")
            raise
    
    def _init_embeddings(self):
        """Inicializar modelo de embeddings"""
        try:
            self.embedder = SentenceTransformer(self.embedding_model)
            print(Fore.GREEN + f"✅ Embeddings: {self.embedding_model}" + Style.RESET_ALL)
        except Exception as e:
            print(Fore.RED + f"❌ Error cargando modelo de embeddings: {e}")
            raise
    
    def _init_llm(self):
        """Inicializar el LLM según el proveedor"""
        try:
            if self.llm_provider == "openai":
                from langchain_openai import ChatOpenAI
                
                if not self.api_key:
                    self.api_key = os.getenv("OPENAI_API_KEY")
                
                if not self.api_key:
                    print(Fore.YELLOW + "⚠️  No se encontró OPENAI_API_KEY, usando modelo local" + Style.RESET_ALL)
                    self.llm_provider = "local"
                    self._init_local_llm()
                else:
                    self.llm = ChatOpenAI(
                        model="gpt-3.5-turbo",
                        api_key=self.api_key,
                        temperature=0.7
                    )
                    print(Fore.GREEN + "✅ LLM: OpenAI GPT-3.5-Turbo" + Style.RESET_ALL)
            
            elif self.llm_provider == "ollama":
                from langchain_community.llms import Ollama
                
                self.llm = Ollama(
                    model="llama2",  # o mistral, codellama, etc.
                    temperature=0.7
                )
                print(Fore.GREEN + "✅ LLM: Ollama (Llama 2)" + Style.RESET_ALL)
            
            elif self.llm_provider == "local":
                self._init_local_llm()
            
            else:
                raise ValueError(f"Proveedor LLM no soportado: {self.llm_provider}")
        
        except ImportError as e:
            print(Fore.YELLOW + f"⚠️  Error de importación: {e}" + Style.RESET_ALL)
            print(Fore.YELLOW + "⚠️  Usando modo sin LLM (solo búsqueda)" + Style.RESET_ALL)
            self.llm = None
    
    def _init_local_llm(self):
        """Inicializar LLM local simple (fallback)"""
        print(Fore.YELLOW + "⚠️  Usando LLM local simple" + Style.RESET_ALL)
        self.llm = None  # Modo sin LLM avanzado
    
    def list_collections(self) -> List[str]:
        """Listar todas las colecciones disponibles"""
        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            print(Fore.RED + f"❌ Error listando colecciones: {e}")
            return []
    
    def get_collection(self, collection_name: str):
        """Obtener una colección específica"""
        try:
            return self.client.get_collection(name=collection_name)
        except Exception as e:
            print(Fore.RED + f"❌ Error obteniendo colección: {e}")
            return None
    
    def semantic_search(self, 
                       query: str, 
                       collection_name: str = None,
                       n_results: int = 5,
                       filters: Optional[Dict] = None) -> List[Dict]:
        """
        Búsqueda semántica en ChromaDB
        
        Args:
            query: Consulta de búsqueda
            collection_name: Nombre de la colección
            n_results: Número de resultados
            filters: Filtros de metadatos
            
        Returns:
            Lista de resultados
        """
        try:
            # Si no se especifica colección, usar la primera disponible
            if not collection_name:
                collections = self.list_collections()
                if not collections:
                    return []
                collection_name = collections[0]
            
            collection = self.get_collection(collection_name)
            if not collection:
                return []
            
            # Realizar búsqueda
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filters,
                include=["documents", "metadatas", "distances"]
            )
            
            # Formatear resultados
            formatted_results = []
            if results["documents"]:
                for i, (doc, metadata, distance) in enumerate(
                    zip(results["documents"][0], 
                        results["metadatas"][0], 
                        results["distances"][0])):
                    
                    formatted_results.append({
                        "content": doc,
                        "metadata": metadata,
                        "distance": distance,
                        "similarity": 1 / (1 + distance) if distance is not None else 1.0,
                        "collection": collection_name,
                        "rank": i + 1
                    })
            
            return formatted_results
        
        except Exception as e:
            print(Fore.RED + f"❌ Error en búsqueda semántica: {e}")
            return []
    
    def generate_answer(self, 
                       query: str, 
                       context: List[Dict] = None,
                       use_history: bool = True) -> Dict:
        """
        Generar respuesta usando LLM con contexto
        
        Args:
            query: Pregunta del usuario
            context: Contexto de búsqueda
            use_history: Usar historial de conversación
            
        Returns:
            Respuesta generada
        """
        # Si no hay LLM, devolver solo resultados de búsqueda
        if self.llm is None:
            return {
                "answer": "Modo de solo búsqueda activado. Aquí están los resultados:",
                "context": context or [],
                "sources": [r["metadata"] for r in context] if context else []
            }
        
        try:
            # Preparar contexto
            context_text = ""
            if context:
                context_text = "\n\nContexto relevante:\n"
                for i, result in enumerate(context[:3]):  # Limitar a 3 resultados
                    context_text += f"\n[{i+1}] {result['content']}\n"
                    if result.get('metadata'):
                        context_text += f"   Fuente: {result['metadata'].get('source', 'N/A')}\n"
            
            # Preparar historial
            history_text = ""
            if use_history and self.conversation_history:
                history_text = "\n\nHistorial de conversación:\n"
                for msg in self.conversation_history[-4:]:  # Últimos 4 mensajes
                    role = "Usuario" if msg["role"] == "user" else "Asistente"
                    history_text += f"{role}: {msg['content']}\n"
            
            # Crear prompt
            prompt = f"""Eres un asistente especializado en analizar y responder preguntas basándote en una base de datos de documentos.

{history_text}

{context_text}

Instrucciones:
1. Responde únicamente en español
2. Si la información en el contexto es relevante, úsala para enriquecer tu respuesta
3. Si el contexto no contiene información relevante, indica que no tienes esa información
4. Sé conciso pero completo
5. Cita las fuentes cuando uses información del contexto

Pregunta del usuario: {query}

Respuesta:"""
            
            # Generar respuesta
            response = self.llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            # Actualizar historial
            self.conversation_history.append({"role": "user", "content": query, "timestamp": datetime.now().isoformat()})
            self.conversation_history.append({"role": "assistant", "content": answer, "timestamp": datetime.now().isoformat()})
            
            # Limitar historial
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            return {
                "answer": answer,
                "context_used": len(context) if context else 0,
                "sources": [r["metadata"] for r in context[:3]] if context else []
            }
        
        except Exception as e:
            print(Fore.RED + f"❌ Error generando respuesta: {e}")
            return {
                "answer": f"Lo siento, hubo un error al generar la respuesta: {str(e)}",
                "context_used": 0,
                "sources": []
            }
    
    def ask_question(self, 
                    question: str, 
                    collection_name: str = None,
                    use_semantic_search: bool = True,
                    n_search_results: int = 3) -> Dict:
        """
        Método principal: preguntar y obtener respuesta
        
        Args:
            question: Pregunta del usuario
            collection_name: Colección a buscar
            use_semantic_search: Realizar búsqueda semántica
            n_search_results: Resultados de búsqueda a usar
            
        Returns:
            Respuesta completa
        """
        print(Fore.YELLOW + f"\n🤔 Pregunta: {question}" + Style.RESET_ALL)
        
        # Paso 1: Búsqueda semántica si está activada
        context = []
        if use_semantic_search:
            print(Fore.CYAN + "🔍 Buscando en la base de datos..." + Style.RESET_ALL)
            context = self.semantic_search(
                query=question,
                collection_name=collection_name,
                n_results=n_search_results
            )
            
            if context:
                print(Fore.GREEN + f"✅ Encontrados {len(context)} documentos relevantes" + Style.RESET_ALL)
                
                # Mostrar resultados de búsqueda
                for i, result in enumerate(context[:2]):
                    preview = result['content'][:150] + "..." if len(result['content']) > 150 else result['content']
                    print(Fore.CYAN + f"   📄 Resultado {i+1} (sim: {result['similarity']:.3f}): {preview}")
            else:
                print(Fore.YELLOW + "⚠️  No se encontraron documentos relevantes" + Style.RESET_ALL)
        
        # Paso 2: Generar respuesta
        print(Fore.CYAN + "💭 Generando respuesta..." + Style.RESET_ALL)
        response = self.generate_answer(query=question, context=context)
        
        return {
            "question": question,
            "answer": response["answer"],
            "search_context": context,
            "sources": response.get("sources", []),
            "timestamp": datetime.now().isoformat()
        }
    
    def analyze_collection(self, collection_name: str) -> Dict:
        """
        Analizar una colección completa
        
        Args:
            collection_name: Nombre de la colección
            
        Returns:
            Análisis de la colección
        """
        try:
            collection = self.get_collection(collection_name)
            if not collection:
                return {"error": "Colección no encontrada"}
            
            # Obtener estadísticas
            count = collection.count()
            
            # Obtener muestras
            results = collection.get(limit=min(10, count))
            
            # Analizar metadatos
            metadata_fields = set()
            if results["metadatas"]:
                for metadata in results["metadatas"]:
                    if metadata:
                        metadata_fields.update(metadata.keys())
            
            # Analizar temas (simple)
            topics = []
            if results["documents"] and self.llm:
                sample_text = "\n".join(results["documents"][:3])
                try:
                    prompt = f"""Analiza estos documentos y extrae los temas principales:

{documentos}

Temas principales (máximo 5, en español):"""
                    
                    response = self.llm.invoke(prompt)
                    topics = [t.strip() for t in response.content.split("\n") if t.strip()]
                except:
                    topics = ["Análisis no disponible"]
            
            return {
                "collection_name": collection_name,
                "document_count": count,
                "metadata_fields": list(metadata_fields),
                "sample_documents": results["documents"][:3] if results["documents"] else [],
                "topics": topics[:5],
                "analysis_date": datetime.now().isoformat()
            }
        
        except Exception as e:
            return {"error": str(e)}
    
    def clear_history(self):
        """Limpiar historial de conversación"""
        self.conversation_history = []
        print(Fore.GREEN + "✅ Historial limpiado" + Style.RESET_ALL)

# Interfaz de línea de comandos mejorada
class ChromaDBAgentCLI:
    def __init__(self, agent: ChromaDBAgent):
        self.agent = agent
        self.running = True
    
    def print_banner(self):
        """Mostrar banner"""
        print(Fore.CYAN + """
╔══════════════════════════════════════════════════════════════╗
║                🤖 AGENTE IA + CHROMADB                       ║
║                 Búsqueda Semántica Avanzada                  ║
╚══════════════════════════════════════════════════════════════╝
""" + Style.RESET_ALL)
    
    def print_help(self):
        """Mostrar ayuda"""
        print(Fore.YELLOW + """
COMANDOS DISPONIBLES:
""" + Style.RESET_ALL)
        print(Fore.GREEN + "  /ask <pregunta>" + Fore.WHITE + "          - Hacer una pregunta")
        print(Fore.GREEN + "  /search <consulta>" + Fore.WHITE + "       - Búsqueda semántica directa")
        print(Fore.GREEN + "  /collections" + Fore.WHITE + "            - Listar colecciones")
        print(Fore.GREEN + "  /analyze <nombre>" + Fore.WHITE + "       - Analizar colección")
        print(Fore.GREEN + "  /history" + Fore.WHITE + "                - Mostrar historial")
        print(Fore.GREEN + "  /clear" + Fore.WHITE + "                  - Limpiar historial")
        print(Fore.GREEN + "  /mode <search|chat|both>" + Fore.WHITE + " - Cambiar modo")
        print(Fore.GREEN + "  /help" + Fore.WHITE + "                   - Mostrar esta ayuda")
        print(Fore.GREEN + "  /exit" + Fore.WHITE + "                   - Salir")
        print()
    
    def print_result(self, result: Dict):
        """Imprimir resultado formateado"""
        print(Fore.GREEN + "\n" + "═" * 60 + Style.RESET_ALL)
        print(Fore.CYAN + "🤖 RESPUESTA:" + Style.RESET_ALL)
        print(Fore.WHITE + result["answer"] + Style.RESET_ALL)
        
        if result.get("sources"):
            print(Fore.YELLOW + "\n📚 FUENTES:" + Style.RESET_ALL)
            for i, source in enumerate(result["sources"][:3]):
                source_name = source.get('filename', source.get('source', 'Documento'))
                print(Fore.WHITE + f"  {i+1}. {source_name}")
        
        if result.get("search_context"):
            print(Fore.MAGENTA + f"\n🔍 Se usaron {len(result['search_context'])} documentos como contexto" + Style.RESET_ALL)
        
        print(Fore.GREEN + "═" * 60 + "\n" + Style.RESET_ALL)
    
    def run(self):
        """Ejecutar interfaz CLI"""
        self.print_banner()
        self.print_help()
        
        current_mode = "both"  # both, search, chat
        current_collection = None
        
        while self.running:
            try:
                # Mostrar prompt
                mode_indicator = {
                    "both": Fore.CYAN + "[BÚSQUEDA+CHAT]",
                    "search": Fore.YELLOW + "[SOLO BÚSQUEDA]",
                    "chat": Fore.MAGENTA + "[SOLO CHAT]"
                }.get(current_mode, "")
                
                prompt_text = f"\n{mode_indicator}{Style.RESET_ALL} {Fore.GREEN}>>{Style.RESET_ALL} "
                user_input = input(prompt_text).strip()
                
                if not user_input:
                    continue
                
                # Comandos especiales
                if user_input.startswith("/"):
                    self._handle_command(user_input, current_mode, current_collection)
                    continue
                
                # Procesar pregunta normal
                if current_mode == "search":
                    # Solo búsqueda semántica
                    results = self.agent.semantic_search(
                        query=user_input,
                        collection_name=current_collection,
                        n_results=5
                    )
                    
                    print(Fore.CYAN + "\n🔍 RESULTADOS DE BÚSQUEDA:" + Style.RESET_ALL)
                    for i, result in enumerate(results):
                        preview = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
                        print(Fore.YELLOW + f"\n[{i+1}] Similitud: {result['similarity']:.3f}")
                        print(Fore.WHITE + f"   {preview}")
                        if result.get('metadata'):
                            source = result['metadata'].get('filename', result['metadata'].get('source', 'N/A'))
                            print(Fore.CYAN + f"   📁 Fuente: {source}")
                
                elif current_mode == "chat":
                    # Solo chat (sin búsqueda)
                    response = self.agent.generate_answer(
                        query=user_input,
                        context=None,
                        use_history=True
                    )
                    
                    print(Fore.CYAN + "\n💭 RESPUESTA:" + Style.RESET_ALL)
                    print(Fore.WHITE + response["answer"] + Style.RESET_ALL)
                
                else:  # both
                    # Búsqueda + respuesta con IA
                    result = self.agent.ask_question(
                        question=user_input,
                        collection_name=current_collection,
                        use_semantic_search=True,
                        n_search_results=3
                    )
                    self.print_result(result)
            
            except KeyboardInterrupt:
                print(Fore.YELLOW + "\n\n👋 Saliendo..." + Style.RESET_ALL)
                self.running = False
            except Exception as e:
                print(Fore.RED + f"❌ Error: {e}" + Style.RESET_ALL)
    
    def _handle_command(self, command: str, current_mode: str, current_collection: str):
        """Manejar comandos especiales"""
        parts = command.split(" ", 1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if cmd == "/exit":
            self.running = False
            print(Fore.YELLOW + "👋 ¡Hasta luego!" + Style.RESET_ALL)
        
        elif cmd == "/help":
            self.print_help()
        
        elif cmd == "/collections":
            collections = self.agent.list_collections()
            if collections:
                print(Fore.CYAN + "\n📚 COLECCIONES DISPONIBLES:" + Style.RESET_ALL)
                for i, col in enumerate(collections):
                    collection = self.agent.get_collection(col)
                    count = collection.count() if collection else 0
                    print(Fore.WHITE + f"  {i+1}. {col} ({count} documentos)")
                
                if current_collection:
                    print(Fore.GREEN + f"\n📌 Colección actual: {current_collection}")
            else:
                print(Fore.YELLOW + "⚠️  No hay colecciones disponibles")
        
        elif cmd == "/analyze":
            if not args:
                print(Fore.RED + "❌ Uso: /analyze <nombre_colección>")
                return
            
            print(Fore.CYAN + f"\n📊 Analizando colección: {args}" + Style.RESET_ALL)
            analysis = self.agent.analyze_collection(args)
            
            if "error" in analysis:
                print(Fore.RED + f"❌ Error: {analysis['error']}")
            else:
                print(Fore.GREEN + f"✅ Documentos: {analysis['document_count']}")
                print(Fore.CYAN + f"📊 Campos de metadatos: {', '.join(analysis['metadata_fields'])}")
                
                if analysis['topics']:
                    print(Fore.YELLOW + "\n🎯 Temas principales:")
                    for topic in analysis['topics']:
                        print(Fore.WHITE + f"  • {topic}")
        
        elif cmd == "/search":
            if not args:
                print(Fore.RED + "❌ Uso: /search <consulta>")
                return
            
            results = self.agent.semantic_search(
                query=args,
                collection_name=current_collection,
                n_results=5
            )
            
            print(Fore.CYAN + f"\n🔍 Resultados para: {args}" + Style.RESET_ALL)
            for i, result in enumerate(results):
                print(Fore.YELLOW + f"\n[{i+1}] Similitud: {result['similarity']:.3f}")
                preview = result['content'][:150] + "..." if len(result['content']) > 150 else result['content']
                print(Fore.WHITE + f"   {preview}")
        
        elif cmd == "/ask":
            if not args:
                print(Fore.RED + "❌ Uso: /ask <pregunta>")
                return
            
            result = self.agent.ask_question(
                question=args,
                collection_name=current_collection,
                use_semantic_search=(current_mode != "chat"),
                n_search_results=3
            )
            self.print_result(result)
        
        elif cmd == "/history":
            history = self.agent.conversation_history
            if history:
                print(Fore.CYAN + "\n📜 HISTORIAL DE CONVERSACIÓN:" + Style.RESET_ALL)
                for i, msg in enumerate(history[-10:]):  # Últimos 10 mensajes
                    role = "👤 Usuario" if msg["role"] == "user" else "🤖 Asistente"
                    color = Fore.BLUE if msg["role"] == "user" else Fore.GREEN
                    print(f"{color}{role}:{Style.RESET_ALL} {msg['content'][:100]}...")
            else:
                print(Fore.YELLOW + "📜 Historial vacío")
        
        elif cmd == "/clear":
            self.agent.clear_history()
        
        elif cmd == "/mode":
            if not args or args not in ["both", "search", "chat"]:
                print(Fore.RED + "❌ Uso: /mode <both|search|chat>")
                return
            
            current_mode = args
            print(Fore.GREEN + f"✅ Modo cambiado a: {current_mode}")
        
        elif cmd == "/use":
            if not args:
                print(Fore.RED + "❌ Uso: /use <nombre_colección>")
                return
            
            current_collection = args
            print(Fore.GREEN + f"✅ Colección seleccionada: {current_collection}")
        
        else:
            print(Fore.RED + f"❌ Comando no reconocido: {cmd}")
            print(Fore.YELLOW + "   Usa /help para ver comandos disponibles")

# Script principal
def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Agente de IA con ChromaDB")
    parser.add_argument("--db-path", default="./chroma_db", help="Ruta de la base de datos ChromaDB")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Modelo de embeddings")
    parser.add_argument("--llm", default="local", choices=["openai", "ollama", "local"], help="Proveedor LLM")
    parser.add_argument("--api-key", help="API key para OpenAI")
    
    args = parser.parse_args()
    
    # Verificar si la base de datos existe
    if not os.path.exists(args.db_path):
        print(Fore.YELLOW + f"⚠️  La base de datos no existe en: {args.db_path}")
        print(Fore.YELLOW + "⚠️  Asegúrate de haber creado primero la base de datos con documentos")
        
        create_anyway = input("¿Continuar de todas formas? (s/n): ").lower().strip()
        if create_anyway != 's':
            print(Fore.YELLOW + "👋 Saliendo...")
            return
    
    # Inicializar agente
    try:
        agent = ChromaDBAgent(
            persist_directory=args.db_path,
            embedding_model=args.model,
            llm_provider=args.llm,
            api_key=args.api_key
        )
        
        # Verificar colecciones
        collections = agent.list_collections()
        if not collections:
            print(Fore.YELLOW + "⚠️  No hay colecciones en la base de datos")
            print(Fore.YELLOW + "⚠️  Primero crea una colección con documentos")
        
        # Iniciar CLI
        cli = ChromaDBAgentCLI(agent)
        cli.run()
    
    except Exception as e:
        print(Fore.RED + f"❌ Error inicializando agente: {e}")
        print(Fore.YELLOW + "\n💡 Posibles soluciones:")
        print(Fore.YELLOW + "1. Instala las dependencias: pip install chromadb sentence-transformers")
        print(Fore.YELLOW + "2. Asegúrate de que el modelo de embeddings existe")
        print(Fore.YELLOW + "3. Verifica los permisos de escritura en el directorio")

# Ejemplo de uso programático
def example_programmatic_use():
    """Ejemplo de uso del agente en código"""
    
    # 1. Inicializar agente
    agent = ChromaDBAgent(
        persist_directory="./mi_base_datos",
        embedding_model="all-MiniLM-L6-v2",
        llm_provider="local"  # Cambiar a "openai" para usar GPT
    )
    
    # 2. Listar colecciones
    collections = agent.list_collections()
    print(f"Colecciones: {collections}")
    
    # 3. Hacer una pregunta
    if collections:
        result = agent.ask_question(
            question="¿Qué es la inteligencia artificial?",
            collection_name=collections[0],  # Usar la primera colección
            use_semantic_search=True,
            n_search_results=3
        )
        
        print(f"Pregunta: {result['question']}")
        print(f"Respuesta: {result['answer'][:500]}...")
        
        if result['sources']:
            print(f"Fuentes: {len(result['sources'])} documentos usados")
    
    # 4. Análisis de colección
    if collections:
        analysis = agent.analyze_collection(collections[0])
        print(f"\nAnálisis de {analysis['collection_name']}:")
        print(f"- Documentos: {analysis['document_count']}")
        print(f"- Campos de metadatos: {analysis['metadata_fields']}")

# Script para cargar documentos iniciales
def create_sample_database():
    """Crear una base de datos de ejemplo si no existe"""
    import chromadb
    from chromadb.config import Settings
    
    print(Fore.CYAN + "📚 Creando base de datos de ejemplo..." + Style.RESET_ALL)
    
    # Crear cliente
    client = chromadb.PersistentClient(
        path="./chroma_db_example",
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Crear colección
    collection = client.create_collection(name="conocimiento_general")
    
    # Documentos de ejemplo
    documents = [
        "La inteligencia artificial es la simulación de procesos de inteligencia humana por máquinas.",
        "Python es un lenguaje de programación interpretado de alto nivel y de propósito general.",
        "Las bases de datos vectoriales almacenan embeddings para búsqueda semántica.",
        "El machine learning es un subcampo de la IA que permite a las máquinas aprender de datos.",
        "ChromaDB es una base de datos vectorial open-source para aplicaciones de IA.",
        "Los transformers son arquitecturas de redes neuronales para procesamiento de lenguaje natural.",
        "Los embeddings son representaciones vectoriales de texto, imágenes u otros datos.",
        "GPT-3 es un modelo de lenguaje grande desarrollado por OpenAI.",
        "La búsqueda semántica entiende el significado de las consultas, no solo palabras clave.",
        "LangChain es un framework para desarrollar aplicaciones con modelos de lenguaje."
    ]
    
    metadatas = [
        {"category": "ai", "topic": "definicion", "source": "wikipedia"},
        {"category": "programming", "topic": "languages", "source": "official_docs"},
        {"category": "databases", "topic": "vector", "source": "tech_blog"},
        {"category": "ai", "topic": "ml", "source": "academic"},
        {"category": "databases", "topic": "tools", "source": "github"},
        {"category": "ai", "topic": "nlp", "source": "research"},
        {"category": "ai", "topic": "embeddings", "source": "tutorial"},
        {"category": "ai", "topic": "llm", "source": "openai"},
        {"category": "search", "topic": "semantic", "source": "article"},
        {"category": "frameworks", "topic": "development", "source": "docs"}
    ]
    
    ids = [f"doc_{i}" for i in range(len(documents))]
    
    # Agregar documentos
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    print(Fore.GREEN + f"✅ Base de datos de ejemplo creada con {len(documents)} documentos")
    print(Fore.CYAN + "📁 Ubicación: ./chroma_db_example")
    print(Fore.YELLOW + "\n💡 Ahora puedes ejecutar: python agent_chroma.py --db-path ./chroma_db_example")

if __name__ == "__main__":
    # Crear base de datos de ejemplo (descomentar si es necesario)
    # create_sample_database()
    
    # Ejecutar agente principal
    main()
    
    # Para uso programático (descomentar):
    # example_programmatic_use()
