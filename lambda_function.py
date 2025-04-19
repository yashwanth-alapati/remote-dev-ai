import json
import urllib3
import os

http = urllib3.PoolManager()

POST_API_URL = "http://54.197.67.80:8080/process"  # <-- replace this

def lambda_handler(event, context):
    # GitHub sends the body as a string
    try:
        body = json.loads(event["body"])
    except Exception as e:
        print(f"Error parsing body: {e}")
        return { "statusCode": 400, "body": "Bad Request" }

    print(body)
    # Basic filter: only respond to new issue comments
    action = body.get("action")
    comment = body.get("comment", {}).get("body", "").strip()
    author_association = body.get("comment", {}).get("author_association", "")
    
    print(f"Received comment: '{comment}' from {author_association}")

    if action == "created" and comment == "/remote-dev" and author_association != "NONE":
        # Trigger your POST request
        print(body)
        try:
            owner = body["repository"]["owner"]["login"]
            repo = body["repository"]["name"]
            issue_heading = body["issue"].get("title", "")
            issue_description = body["issue"].get("body", "")
            data = {
                "owner_name": owner,
                "repo_name": repo,
                "issue_number": str(body["issue"]["number"]),
                "issue_description": issue_heading + " : " + issue_description,
                "commenter": body["comment"]["user"]["login"]
            }
            
            print(f"POST request data: {data}")
            r = http.request(
                "POST",
                POST_API_URL,
                body=json.dumps(data).encode("utf-8"),
                headers={ "Content-Type": "application/json" }
            )
            print(f"POST response: {r.status}")
        except Exception as e:
            print(f"POST request failed: {e}")
            return { "statusCode": 500, "body": "Internal error" }

    return { "statusCode": 200, "body": "OK" }
