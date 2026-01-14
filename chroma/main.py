# main.py - Ejemplo completo de uso
from chroma_vector_db import *


def main():
    print("=" * 60)
    print("BASE DE DATOS VECTORIAL CON CHROMADB")
    print("=" * 60)
    
    # 1. Inicializar la base de datos
    print("\n1. Inicializando base de datos...")
    db = ChromaVectorDatabase(
        collection_name="my_documents",
        persist_directory="./chroma_storage",
        embedding_model="all-MiniLM-L6-v2"  # None para usar embeddings por defecto
    )
    
    # 2. Crear algunos documentos de ejemplo
    print("\n2. Creando documentos de ejemplo...")
    
    documents = [
        "Python es un lenguaje de programación interpretado de alto nivel.",
        "Las bases de datos vectoriales permiten búsqueda semántica.",
        "ChromaDB es una base de datos vectorial open-source.",
        "Los embeddings transforman texto en vectores numéricos.",
        "El machine learning usa algoritmos para aprender de los datos.",
        "Las redes neuronales son la base del deep learning.",
        "El procesamiento de lenguaje natural (NLP) analiza texto humano.",
        "Los transformers revolucionaron el campo del NLP."
    ]
    
    metadatas = [
        {"category": "programming", "language": "python", "topic": "languages"},
        {"category": "databases", "type": "vector", "topic": "search"},
        {"category": "databases", "type": "vector", "topic": "tools"},
        {"category": "ai", "concept": "embeddings", "topic": "representation"},
        {"category": "ai", "field": "ml", "topic": "algorithms"},
        {"category": "ai", "field": "deep learning", "topic": "neural networks"},
        {"category": "ai", "field": "nlp", "topic": "text analysis"},
        {"category": "ai", "field": "nlp", "topic": "architecture"}
    ]
    
    # 3. Agregar documentos a la colección
    db.add_documents(documents, metadatas)
    
    # 4. Realizar búsquedas
    print("\n3. Realizando búsquedas...")
    
    queries = [
        "lenguajes de programación",
        "bases de datos de vectores",
        "aprendizaje automático"
    ]
    
    for query in queries:
        print(f"\n🔍 Búsqueda: '{query}'")
        print("-" * 40)
        
        results = db.search(query, n_results=3)
        
        for result in results:
            print(f"📄 [{result['rank']}] Sim: {result['similarity']:.3f}")
            print(f"   {result['document']}")
            print(f"   📊 Categoría: {result['metadata'].get('category', 'N/A')}")
            print()
    
    # 5. Búsqueda con filtros
    print("\n4. Búsqueda con filtros...")
    print("\n🔍 Búsqueda: 'aprendizaje' (solo categoría 'ai')")
    print("-" * 40)
    
    filtered_results = db.search(
        "aprendizaje",
        n_results=3,
        where_filter={"category": "ai"}
    )
    
    for result in filtered_results:
        print(f"[{result['rank']}] {result['document'][:80]}...")
    
    # 6. Información de la colección
    print("\n5. Información de la colección:")
    print("-" * 40)
    
    info = db.get_collection_info()
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # 7. Ejemplo con archivos reales
    print("\n6. Ejemplo con archivos de texto:")
    print("-" * 40)
    
    # Crear un archivo de ejemplo
    sample_text = """La inteligencia artificial (IA) es la simulación de procesos 
    de inteligencia humana por parte de máquinas, especialmente sistemas informáticos. 
    Estos procesos incluyen el aprendizaje, el razonamiento y la autocorrección."""
    
    with open("sample_ai.txt", "w", encoding="utf-8") as f:
        f.write(sample_text)
    
    # Procesar el archivo
    processor = DocumentProcessor()
    loaded_docs = processor.load_directory(".", extensions=['.txt'])
    
    if loaded_docs:
        texts = [doc["text"] for doc in loaded_docs]
        metadata_list = [doc["metadata"] for doc in loaded_docs]
        
        db.add_documents(texts, metadata_list)
        
        # Buscar en el nuevo contenido
        print("\n🔍 Búsqueda en archivos cargados: 'inteligencia artificial'")
        results = db.search("inteligencia artificial", n_results=2)
        
        for result in results:
            print(f"📄 {result['document'][:100]}...")
            print(f"   📁 Archivo: {result['metadata'].get('filename', 'N/A')}")
    
    # Limpiar archivo de ejemplo
    if os.path.exists("sample_ai.txt"):
        os.remove("sample_ai.txt")
    
    print("\n" + "=" * 60)
    print("✅ Base de datos vectorial creada exitosamente")
    print(f"📁 Datos guardados en: {db.persist_directory}")
    print("=" * 60)

# Ejemplo avanzado: Interfaz interactiva
def interactive_example():
    """Ejemplo con interfaz interactiva"""
    import json
    
    db = ChromaVectorDatabase(
        collection_name="interactive_docs",
        persist_directory="./interactive_chroma"
    )
    
    while True:
        print("\n" + "=" * 50)
        print("MENÚ BASE DE DATOS VECTORIAL")
        print("=" * 50)
        print("1. Agregar documentos")
        print("2. Buscar documentos")
        print("3. Ver información de colección")
        print("4. Salir")
        
        choice = input("\nSeleccione una opción: ").strip()
        
        if choice == "1":
            print("\n📝 AGREGAR DOCUMENTOS")
            print("(Escriba 'fin' en una línea vacía para terminar)")
            
            documents = []
            metadatas = []
            
            while True:
                print(f"\nDocumento #{len(documents) + 1}")
                doc = input("Texto del documento: ").strip()
                
                if doc.lower() == 'fin':
                    break
                
                if not doc:
                    print("⚠️ Documento vacío, omitiendo...")
                    continue
                
                # Metadatos
                print("Metadatos (formato JSON simple, ej: {\"categoria\": \"ciencia\"})")
                metadata_str = input("Metadatos (opcional): ").strip()
                
                metadata = {}
                if metadata_str:
                    try:
                        metadata = json.loads(metadata_str)
                    except:
                        print("⚠️ Error en formato JSON, usando metadatos vacíos")
                
                documents.append(doc)
                metadatas.append(metadata)
            
            if documents:
                db.add_documents(documents, metadatas)
                print(f"✅ {len(documents)} documentos agregados")
        
        elif choice == "2":
            print("\n🔍 BUSCAR DOCUMENTOS")
            query = input("Consulta de búsqueda: ").strip()
            
            if not query:
                print("⚠️ Consulta vacía")
                continue
            
            try:
                n_results = int(input("Número de resultados (default 5): ") or "5")
            except:
                n_results = 5
            
            results = db.search(query, n_results=n_results)
            
            print(f"\n📊 Resultados para: '{query}'")
            print("-" * 60)
            
            if not results:
                print("No se encontraron resultados")
            else:
                for result in results:
                    print(f"\n[{result['rank']}] Similitud: {result['similarity']:.3f}")
                    print(f"📄 {result['document'][:150]}...")
                    if result['metadata']:
                        print(f"📊 Metadatos: {json.dumps(result['metadata'], ensure_ascii=False)[:100]}...")
        
        elif choice == "3":
            info = db.get_collection_info()
            print("\n📊 INFORMACIÓN DE COLECCIÓN")
            print("-" * 40)
            for key, value in info.items():
                print(f"{key}: {value}")
        
        elif choice == "4":
            print("\n👋 Saliendo...")
            break
        
        else:
            print("⚠️ Opción no válida")

if __name__ == "__main__":
    # Ejecutar ejemplo principal
    main()
    
    # Para ejecutar la interfaz interactiva:
    #interactive_example()
