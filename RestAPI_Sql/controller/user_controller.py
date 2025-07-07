from app import app
from model.user_model import user_model
from model.auth_model import auth_model
from flask import request, send_file
from datetime import datetime

obj = user_model()
auth = auth_model()

#@app.route("/user/signup")
#def signup():
#    return "This is signup operation"

#def user_signup_controller():
#    return obj.user_signup_model()

@app.route("/user/getall")
@auth.token_auth()
def user_getall_controller():
    return obj.user_getall_model()

@app.route("/user/addone", methods=["POST"])
@auth.token_auth()
def user_addone_controller():
    data = request.form
    return obj.user_addone_model(data)

@app.route("/user/addmultiple", methods=["POST"])
@auth.token_auth()
def user_addmultiple_controller():
    data = request.json
    return obj.user_addmultiple_model(data)

@app.route("/user/update", methods=["PUT"])
def user_update_controller():
    data = request.form
    return obj.user_update_model(data)

@app.route("/user/delete/<id>", methods=["DELETE"])
def user_delete_controller():
    data = request.form
    return obj.user_delete_model(data)

@app.route("/user/patch/<id>", methods=["PATCH"])
def user_patch_controller():
    data = request.form
    return obj.user_patch_model(data, data)

# Pagination
@app.route("/user/getall/limit/<limit>/page/<page>", methods=["GET"])
def user_pagination_controller(limit, page):
    return obj.user_pagination_model(limit, page)

# 1. Upload file from postman t server
# 2. Saving the file into te file system with unique filename
# 3. filepath in database with respective entity
@app.route("/user/<uid>/upload/avatar", methods=["PUT"])
def user_upload_avatar_controller(uid):
    file = request.files['avatar']
    uniqueFileName = str(datetime.now().timestamp()).repalce(".", "")
    fileNameSplit = file.filename.split(".")
    ext = fileNameSplit[len(fileNameSplit)-1]
    saveFilePath = f"uploads/{uniqueFileName}.{ext}"
    file.save(saveFilePath)
    return obj.user_upload_avatar_model(uid, saveFilePath)

# Creating an endpoint to read the file
@app.route("/uploads/<filename>")
def user_getavatar_controller(filename):
    return send_file(f"uploads/{filename}")

# JWT endpoint
@app.route("/user/login", methods=["POST"])
def user_login_controller():
    data = request.form
    return obj.user_login_model(data)