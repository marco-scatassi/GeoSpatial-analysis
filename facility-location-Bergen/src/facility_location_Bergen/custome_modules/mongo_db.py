from pymongo import MongoClient, database

# ----------------------------------------------------- get database -----------------------------------------------------
def get_database(db_name, connection_string):
 
   # Create a connection using MongoClient. You can import MongoClient or use pymongo.MongoClient
   client = MongoClient(connection_string)
   # Create the database for our example (we will use the same database throughout the tutorial
   return client[db_name]

# ----------------------------------------------------- get collection -----------------------------------------------------
def retrieve_collections_names(db: database.Database, name: str, days: list):
    collections_names = [name+"_data_"+day for day in days]
    return collections_names

# ----------------------------------------------- get database and collection ----------------------------------------------
def retrieve_database_and_collections(db_name: str, days: list, which: list, connection_string="mongodb://localhost:27017"):
   if isinstance(days, list) and isinstance(which, list):
      pass
   else:
      raise TypeError('days and/or which must be a list')

   # Get the database
   db = get_database(db_name, connection_string)
   # define the collections names
   collections_names = []
   for name in which:
      collections_names += retrieve_collections_names(db, name, days)
   # Get the collections
   collections = {collection_name:db[collection_name] for collection_name in collections_names}
   return db, collections