FROM --platform=${BUILDPLATFORM} python:3.9.16

WORKDIR /app

# Add the monitor user.
RUN \
    groupadd -g 1001 monitor \
    && useradd --no-log-init --create-home --shell /bin/bash --gid 1001 --uid 1001 monitor

USER monitor

COPY --chown=monitor:monitor requirements.txt /app
# Install Python packages.
RUN --mount=type=cache,mode=0755,target=/home/monitor/.cache/pip \
    pip3 install -r requirements.txt
ENV PATH="${PATH}:/home/monitor/.local/bin"

COPY --chown=monitor:monitor . /app

COPY docker-entrypoint.sh /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]