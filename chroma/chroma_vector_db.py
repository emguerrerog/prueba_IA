# chroma_vector_db.py
import os
import glob
import chromadb
from typing import List, Dict, Optional, Union
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
import uuid
from datetime import datetime
from chromadb.utils import embedding_functions

class ChromaVectorDatabase:
    def __init__(self, 
                 collection_name: str = "documents",
                 persist_directory: str = "./chroma_db",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Inicializa la base de datos vectorial con ChromaDB
        
        Args:
            collection_name: Nombre de la colección
            persist_directory: Directorio para persistencia
            embedding_model: Modelo para embeddings (o None para usar embeddings de Chroma)
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Configurar ChromaDB
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Usar embeddings personalizados si se especifica un modelo
        if embedding_model:
            self.embedding_model = SentenceTransformer(embedding_model)
            #self.embedding_function = self.custom_embedding_function
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        else:
            self.embedding_model = None
            self.embedding_function = None
        
        # Crear o obtener la colección
        self.collection = self.get_or_create_collection()
    
    def custom_embedding_function(self, texts: List[str]) -> List[List[float]]:
        """Función personalizada para generar embeddings"""
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def get_or_create_collection(self):
        """Obtiene o crea una colección"""
        try:
            # Intentar obtener la colección existente
            collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            print(f"Colección '{self.collection_name}' cargada")
        except:
            # Crear nueva colección
            collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"created": datetime.now().isoformat()}
            )
            print(f"Nueva colección '{self.collection_name}' creada")
        
        return collection
    
    def add_documents(self, 
                     documents: List[str],
                     metadatas: Optional[List[Dict]] = None,
                     ids: Optional[List[str]] = None):
        """
        Agrega documentos a la colección
        
        Args:
            documents: Lista de textos/documentos
            metadatas: Metadatos para cada documento
            ids: IDs únicos para cada documento (auto-generados si no se proporcionan)
        """
        if not documents:
            raise ValueError("La lista de documentos no puede estar vacía")
        
        # Generar IDs si no se proporcionan
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        
        # Preparar metadatos
        if metadatas is None:
            metadatas = [{} for _ in documents]
        
        # Asegurar que cada metadata tenga timestamp
        for i, metadata in enumerate(metadatas):
            if "added_date" not in metadata:
                metadata["added_date"] = datetime.now().isoformat()
            if "document_length" not in metadata:
                metadata["document_length"] = len(documents[i])
        
        # Agregar a la colección
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✅ {len(documents)} documentos agregados a la colección '{self.collection_name}'")
    
    def search(self, 
               query: str, 
               n_results: int = 5,
               where_filter: Optional[Dict] = None,
               where_document_filter: Optional[Dict] = None) -> List[Dict]:
        """
        Busca documentos similares
        
        Args:
            query: Texto de consulta
            n_results: Número de resultados
            where_filter: Filtro por metadatos
            where_document_filter: Filtro por contenido del documento
            
        Returns:
            Lista de resultados con documentos, metadatos y distancias
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter,
            where_document=where_document_filter,
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
                    "id": results["ids"][0][i] if results["ids"] else None,
                    "document": doc,
                    "metadata": metadata,
                    "distance": distance,
                    "similarity": 1 / (1 + distance) if distance is not None else None,
                    "rank": i + 1
                })
        
        return formatted_results
    
    def update_document(self, 
                       document_id: str, 
                       new_document: str = None,
                       new_metadata: Dict = None):
        """
        Actualiza un documento existente
        """
        if new_document:
            self.collection.update(
                ids=[document_id],
                documents=[new_document]
            )
        
        if new_metadata:
            self.collection.update(
                ids=[document_id],
                metadatas=[new_metadata]
            )
    
    def delete_documents(self, ids: List[str] = None, where_filter: Dict = None):
        """
        Elimina documentos
        """
        self.collection.delete(
            ids=ids,
            where=where_filter
        )
    
    def get_collection_info(self) -> Dict:
        """Obtiene información de la colección"""
        count = self.collection.count()
        
        return {
            "name": self.collection_name,
            "document_count": count,
            "persist_directory": self.persist_directory,
            "embedding_model": "custom" if self.embedding_model else "default"
        }
    
    def list_collections(self):
        """Lista todas las colecciones"""
        return self.client.list_collections()
    
    def reset_collection(self):
        """Elimina y recrea la colección"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.get_or_create_collection()
        print(f"Colección '{self.collection_name}' reiniciada")

# Procesador de documentos
class DocumentProcessor:
    @staticmethod
    def load_text_file(filepath: str) -> str:
        """Carga texto desde archivo"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    
    @staticmethod
    def split_text(text: str, 
                   chunk_size: int = 500, 
                   chunk_overlap: int = 50) -> List[str]:
        """
        Divide texto en chunks
        
        Args:
            text: Texto a dividir
            chunk_size: Tamaño máximo del chunk
            chunk_overlap: Superposición entre chunks
            
        Returns:
            Lista de chunks
        """
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        return text_splitter.split_text(text)
    
    @staticmethod
    def load_directory(directory_path: str, 
                      extensions: List[str] = ['.txt', '.md', '.pdf']) -> List[Dict]:
        """
        Carga todos los archivos de un directorio
        
        Args:
            directory_path: Ruta del directorio
            extensions: Extensiones de archivo a incluir
            
        Returns:
            Lista de documentos con metadatos
        """
        documents = []
        
        for ext in extensions:
            pattern = os.path.join(directory_path, f"*{ext}")
            files = glob.glob(pattern)
            
            for filepath in files:
                try:
                    content = DocumentProcessor.load_text_file(filepath)
                    
                    metadata = {
                        "source": filepath,
                        "filename": os.path.basename(filepath),
                        "extension": ext,
                        "file_size": os.path.getsize(filepath),
                        "last_modified": datetime.fromtimestamp(
                            os.path.getmtime(filepath)).isoformat()
                    }
                    
                    # Dividir en chunks si es muy largo
                    if len(content) > 1000:
                        chunks = DocumentProcessor.split_text(content)
                        for i, chunk in enumerate(chunks):
                            chunk_metadata = metadata.copy()
                            chunk_metadata["chunk"] = i
                            chunk_metadata["total_chunks"] = len(chunks)
                            documents.append({
                                "text": chunk,
                                "metadata": chunk_metadata
                            })
                    else:
                        documents.append({
                            "text": content,
                            "metadata": metadata
                        })
                    
                except Exception as e:
                    print(f"Error procesando {filepath}: {e}")
        
        return documents

