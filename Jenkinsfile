pipeline {
    agent any

    stages {

        stage('Install Dependencies') {
            steps {
                bat 'pip install -r requirements.txt'
            }
        }

        stage('Train Model') {
            steps {
                bat 'python src/train.py'
            }
        }

        stage('Show Metrics') {
            steps {
                bat 'type metrics.json'
            }
        }

    }
}