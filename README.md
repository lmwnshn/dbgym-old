# HAKUNA MATATA


Quick notes on design

- Optimize for student-level simplicity, not best practice.
- All in on docker compose and microservices.
- Each microservice should keep private state in its own container.
- State can be shared through the gym_db.
- Communicate via REST API, reserve names and ports in the .env file.


Docker...

- Watch out for shared memory.