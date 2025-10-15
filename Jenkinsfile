pipeline {
    agent any // Specifies where the pipeline will execute

    stages {
        // stage('Checkout') {
        //     steps {
        //         // Get the code from your Git repository
        //         git 'https://github.com/aman123443/stress-level-app.git'
        //         echo 'Code checked out successfully.'
        //     }
        // }

        stage('Build Docker Image') {
            steps {
                script {
                    // Define a variable for the image name
                    def imageName = 'my-app:latest'
                    echo "Building Docker image: ${imageName}"
                    
                    // Build the Docker image using the Dockerfile in the current directory
                    docker.build(imageName, '.')
                }
            }
        }

        stage('Deploy Container') {
            steps {
                script {
                    def imageName = 'my-app:latest'
                    def containerName = 'my-running-app'

                    echo "Deploying container: ${containerName}"
                    
                    // Check if a container with the same name is running and stop/remove it
                    sh "docker ps -a | grep ${containerName} && docker stop ${containerName} && docker rm ${containerName} || true"
                    
                    // Run the new container
                    // -d: run in detached mode
                    // -p 8080:3000: map port 8080 on the host to port 3000 in the container
                    // --name: give the container a name
                    sh "docker run -d -p 8080:3000 --name ${containerName} ${imageName}"
                }
            }
        }
    }

    post {
        always {
            // Clean up old Docker images to save space
            echo 'Cleaning up old Docker images...'
            sh 'docker image prune -f'
        }
    }
}