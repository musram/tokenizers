# Use the official Rust image as a parent image
FROM rust:1.70-slim-buster

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container
COPY . .

# Build the application
RUN cargo build --release

# Run the binary
#CMD ["./target/release/your_binary_name"]