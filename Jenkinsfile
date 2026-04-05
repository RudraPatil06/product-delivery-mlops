pipeline {
    agent any

    environment {
        IMAGE_NAME = "delivery-mlops"
        CONTAINER_NAME = "delivery-container"
    }

    stages {

        stage('📥 Clone Code') {
            steps {
                echo "Code pulled from GitHub automatically"
                bat 'dir'
            }
        }

        stage('🧹 Clean Old Container') {
    steps {
        bat '''
        docker stop %CONTAINER_NAME% >nul 2>&1 || exit 0
        docker rm %CONTAINER_NAME% >nul 2>&1 || exit 0
        '''
    }
}

        stage('📂 Check Files') {
            steps {
                bat 'dir'
            }
        }

        stage('🐳 Build Docker Image') {
            steps {
                bat 'docker build -t %IMAGE_NAME% .'
            }
        }

        stage('🚀 Run Container') {
            steps {
                bat '''
                docker run -d -p 8501:8501 -p 8000:8000 --name %CONTAINER_NAME% %IMAGE_NAME%
                '''
            }
        }

        stage('📊 Check Running') {
            steps {
                bat 'docker ps'
            }
        }
    }

    post {
        success {
            echo "✅ SUCCESS"
            echo "🌐 Dashboard: http://localhost:8501"
            echo "📡 API: http://localhost:8000/docs"
        }
        failure {
            echo "❌ FAILED - Check logs"
        }
    }
}