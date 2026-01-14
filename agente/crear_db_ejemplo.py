import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(path='./mi_chroma_db')
collection = client.create_collection(name='documentos')

# Agregar documentos
collection.add(
    documents=['La IA transforma industrias', 'Python es versátil'],
    metadatas=[{'fuente': 'web'}, {'fuente': 'docs'}],
    ids=['doc1', 'doc2']
)
print('✅ Base de datos creada')
