import mysql.connector
import json
from flask import make_response
from datetime import datetime, timedelta
import jwt
from configs.config import dbconfig

class user_model():

    def __init__(self):
        # Connection establishment code
        try:
            self.cn = mysql.connector.connect(host=dbconfig['host'],user=dbconfig['username'],password=dbconfig['password'],database=dbconfig['database'])
            self.cn.autocommit=True
            self.cur = self.cn.cursor(dictionary=True)
            print("Connection Successful")
        except:
            print("Some error")

    def user_signup_model(self):
        # Business Logic
        return "This is user_signup_model"
    
    def user_getall_model(self):
        # Query execution code
        self.cur.execute("Select * FROM users")
        res = self.cur.fetchall()
        if len(res)>0:
            #return json.dumps(res) # json as dict
            #return {"Payload":res} # list as dict, direct list is not allowed.
            #return make_response({"Payload":res}, 200) # with response code
            res = make_response({"Payload":res}, 200)
            res.headers["Access-Control-Allow-Origin"]="*"
            return res
        else:
            return make_response({"message":"No Data Found"}, 204)
        
    def user_addone_model(self, data):
        # Query execution code
        self.cur.execute(f"INSERT into users(name, email, phone, role, password) VALUES('{data['name']}', '{data['email']}', '{data['name']}', '{data['phone']}', '{data['role']}, '{data['password']}'')")
        return make_response({"message":"User created sucessfully"}, 201)
    
    def user_addmultiple_model(self, data):
        # Query execution 
        query = "INSERT into users(name, email, phone, role_id, password) VALUES "
        for userdata in data:
            query += f"('{userdata['name']}, {userdata['email']}, {userdata['phone']}, {userdata['role_id']}, {userdata['password']})"
        finalQuery = query.rstrip(",")
        self.cur.execute(finalQuery)
        return make_response({"message":"Multiple Users created sucessfully"}, 201)
    
    def user_update_model(self, data):
        # Query execution code
        self.cur.execute("UPDATE users SET name='{data['name']}', email='{data['email']}', phone='{data['phone']}', role='{data['role']}', password='{data['password']}' WHERE id='{data['id']}'")
        if self.cur.rowcount>0:
            return make_response({"message":"User updated sucessfully"}, 201)
        else:
            return make_response({"message":"Nothing to udpate"}, 202)
        
    def user_delete_model(self, id):
        # Query execution code
        self.cur.execute("DELETE from users where id={id}")
        if self.cur.rowcount>0:
            return make_response({"message":"User deleted sucessfully"}, 200)
        else:
            return make_response({"message":"Nothing to delete"}, 202)
        
    def user_patch_model(self, data, id):
        query = "UPDATE users SET "
        for key in data:
            query += f"{key}='{data[key]}',"
        query = query[:-1] + f"WHERE id={id}"
        self.cur.execute(query)
        if self.cur.rowcount>0:
            return make_response({"message":"User updated sucessfully"}, 201)
        else:
            return make_response({"message":"Nothing to udpate"}, 202)
        
    def user_pagination_model(self, limit, page):
        limit = int(limit)
        page = int(page)
        start = (page*limit)-limit
        query = f"SELECT * from users LIMIT {start}, {limit}"
        self.cur.execute(query)
        res = self.cur.fetchall()
        if len(res)>0:
            #return json.dumps(res) # json as dict
            #return {"Payload":res} # list as dict, direct list is not allowed.
            #return make_response({"Payload":res}, 200) # with response code
            res = make_response({"Payload":res, "page_no":page, "limit":limit}, 200)
            res.headers["Access-Control-Allow-Origin"]="*"
            return res
        else:
            return make_response({"message":"No Data Found"}, 204)

    def user_uplaod_avatar_model(self, uid, filepath):
        self.cur.execute(f"UPDATE users SET avatar='{filepath}' WHERE id={uid}")
        if self.cur.rowcount>0:
            return make_response({"message":"File uploaded successfully"}, 201)
        else:
            return make_response({"message":"Nothing to udpate"}, 202)

    def user_login_model(self, data):
        self.cur.execute(f"SELECT id, name, name, email, phone, avatar, role_id FROM users where email='{data["email"]}' and password='{data["password"]}'")
        res = self.cur.fetchall()
        userdata = res[0] # First entry for id
        exp_time = datetime.now() + timedelta(minutes=15)
        exp_epoch_time = int(exp_time.timestamp())
        payload = {
            "payload":userdata,
            "exp":exp_epoch_time # key exp is compulsory for jwt
        }
        jwtoken = jwt.encode(payload, "anand", algorithm="HS256") # Check algorithm types on jwt.io
        return make_response({"token":jwtoken}, 200)