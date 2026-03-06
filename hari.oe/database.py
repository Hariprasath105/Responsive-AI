database = {
    "user1": {"name": "Alice", "data": "..."},
    "user2": {"name": "Bob", "data": "..."},
    "user3": {"name": "ram", "data": "..."},
    "user4": {"name": "jeeva", "data": "..."},
    "user5": {"name": "Mani", "data": "..."}
}
def delete_user_data(user_id):
    if user_id in database:
        del database[user_id]
        print(f"User {user_id} data deleted.")
    else:
        print("User not found.")
delete_user_data("user2")
print(database)
