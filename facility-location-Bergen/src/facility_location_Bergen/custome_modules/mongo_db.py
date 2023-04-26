from pymongo import MongoClient, database

# ----------------------------------------------------- get database -----------------------------------------------------
def get_database(db_name, connection_string):
 
   # Create a connection using MongoClient. You can import MongoClient or use pymongo.MongoClient
   client = MongoClient(connection_string)
   # Create the database for our example (we will use the same database throughout the tutorial
   return client[db_name]

# ----------------------------------------------------- get collection -----------------------------------------------------
def retrieve_collections_names(db: database.Database, name: str, day: str):
    collections_names = name+"_data_"+day 
    return collections_names

# ----------------------------------------------- get database and collection ----------------------------------------------
def retrieve_database_and_collections(db_name: str, day: str, which: list, connection_string="mongodb://localhost:27017"):
   # Get the database
   db = get_database(db_name, connection_string)
   # define the collections names
   collections_names = []
   for name in which:
      collections_names += [retrieve_collections_names(db, name, day)]
   # Get the collections
   collections = {collection_name:db[collection_name] for collection_name in collections_names}
   return db, collections

# ------------------------------------------------- get only empty collection ------------------------------------------------
def take_empty_collections(collections: dict):
    empty_collections = {}
    for collection_name, collection in collections.items():
        if collection.count_documents({}) == 0:
            empty_collections[collection_name] = collection
    return empty_collections