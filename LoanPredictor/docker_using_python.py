import docker

# Create a Docker client
client = docker.from_env()

# Specify the Docker image you want to run
image_name = 'anandkhare001/loan-pred'  # Replace with your image name
container_name = 'loan-app'  # Optional: Replace with your desired container name

try:
    # Pull the image (if not already pulled)
    client.images.pull(image_name)

    # Run the container
    container = client.containers.run(image_name, name=container_name, ports={5000: None}, detach=True)

    print(f'Container {container.name} is running with ID: {container.id}')

    # Optional: You can wait for the container to finish, get logs, etc.
    # container.wait()
    # logs = container.logs()
    # print(logs.decode())

except docker.errors.ImageNotFound:
    print(f'Image {image_name} not found.')
except docker.errors.APIError as e:
    print(f'Error: {e}')

# Clean up: Stop and remove the container if needed
# container.stop()
# container.remove()
