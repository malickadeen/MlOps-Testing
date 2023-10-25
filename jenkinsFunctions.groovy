def checkoutCode(String repoUrl, String branch) {
    checkout([$class: 'GitSCM', branches: [[name: branch]], doGenerateSubmoduleConfigurations: false, extensions: [], submoduleCfg: [], userRemoteConfigs: [[url: repoUrl]]])
}

def buildProject() {
    echo 'Building'
    bat 'pip3 install -r requirements.txt'
}

def runTests() {
    echo 'Test'
    bat 'python test.py'
}

def deployApplication() {
    echo 'Deploy'
}

return this
