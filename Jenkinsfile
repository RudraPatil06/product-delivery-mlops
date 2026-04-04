pipeline {
    agent any

    environment {
        PROJECT_NAME = "delivery-mlops"
        IMAGE_NAME = "delivery-app"
    }

    stages {

        stage('1. Clone Repository') {
            steps {
                echo 'Cloning GitHub repository...'
                git 'https://github.com/your-username/your-repo.git'
            }
        }

        stage('2. Install Dependencies') {
            steps {
                echo 'Installing dependencies...'
                bat 'pip install -r requirements.txt'
            }
        }

        stage('3. Data Validation') {
            steps {
                echo 'Checking dataset...'
                bat 'dir data'
            }
        }

        stage('4. Train Model') {
            steps {
                echo 'Training model...'
                bat 'python src/train.py'
            }
        }

        stage('5. Model Evaluation') {
            steps {
                echo 'Checking metrics...'
                bat 'type metrics.json'
            }
        }

        stage('6. MLflow Tracking') {
            steps {
                echo 'MLflow already tracking experiment inside training script'
            }
        }

        stage('7. DVC Data Versioning') {
            steps {
                echo 'Tracking data with DVC...'
                bat 'python -m dvc add data/delivery_data.csv'
                bat 'python -m dvc add delivery_time_model.pkl'
            }
        }

        stage('8. Build Docker Image') {
            steps {
                echo 'Building Docker image...'
                bat 'docker build -t %IMAGE_NAME% .'
            }
        }

        stage('9. Run Container') {
            steps {
                echo 'Deploying application...'
                bat 'docker run -d -p 8501:8501 %IMAGE_NAME%'
            }
        }

        stage('10. Post Deployment Check') {
            steps {
                echo 'Application deployed at http://localhost:8501'
            }
        }
    }

    post {
        success {
            echo 'Pipeline executed successfully 🎉'
        }
        failure {
            echo 'Pipeline failed ❌'
        }
    }
}