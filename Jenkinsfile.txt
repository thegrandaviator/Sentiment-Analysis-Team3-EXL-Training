pipeline {  
    agent any
    environment {
        CONDA_ENV = 'sentiment_env'
    }
    stages {
        stage('Checkout SCM') {
            steps {
                git url: 'https://github.com/Pratheekheb/Sentimentanalysisproject.git', branch: 'main', credentialsId: '5f06b98e-537d-4c25-bef5-be1a4bfd66ac'
            }
        }

        stage('Install Dependencies') {
            steps {
                script {
                    bat 'conda create -n ' + CONDA_ENV + ' python=3.8 -y'
                }
            }
        }

        stage('Activate Environment') {
            steps {
                script {
                    bat 'conda activate ' + CONDA_ENV
                }
            }
        }

        stage('Install Required Packages') {
            steps {
                script {
                    // Modify requirements.txt to ensure compatibility with Python 3.8
                    bat """
                    conda activate ${CONDA_ENV} && \
                    echo streamlit>=1.37.1 > temp_requirements.txt && \
                    echo scikit-learn>=1.1.2 >> temp_requirements.txt && \
                    echo nltk>=3.9.1 >> temp_requirements.txt && \
                    echo beautifulsoup4>=4.12.2 >> temp_requirements.txt && \
                    echo numpy==1.24.4 >> temp_requirements.txt && \
                    echo pandas==1.5.3 >> temp_requirements.txt && \
                    echo wordcloud>=1.9.3 >> temp_requirements.txt && \
                    pip install -r temp_requirements.txt
                    """
                }
            }
        }

        stage('Run Sentiment Analysis Model') {
            steps {
                script {
                    bat 'conda activate ' + CONDA_ENV + ' && python model.py'
                }
            }
        }
    }

    post {
        always {
            cleanWs()
        }
        failure {
            echo 'Pipeline failed!'
        }
        success {
            echo 'Pipeline succeeded!'
        }
    }
}
