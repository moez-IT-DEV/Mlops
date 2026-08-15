from dagshub.upload import Repo
import git
USERNAME="moezkchaoumail"
REPONAME="Mlops"
COMMIT="bestModel2"

repo = git.Repo(".")
last_commit = repo.head.commit
#print(f"last_commit={last_commit}")

#Convert from class git.commit.commit to str
last_commit=str(last_commit)

repo = Repo(owner=USERNAME,name=REPONAME,username=USERNAME,password="moez123.",branch="main")  # Optional: username, password, token, branch
# Upload a single file to a repository in one line
LOCALFILEPATH="./backend/models/best_model_2.pkl"
PATHINREMOTE="./backend/models/best_model_3.pkl"
repo.upload(file=LOCALFILEPATH, path=PATHINREMOTE, versioning="git",commit_message=COMMIT,last_commit=last_commit)  # Optional: versioning, new_branch, commit_message
