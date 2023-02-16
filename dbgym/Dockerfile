FROM --platform=${BUILDPLATFORM} python:3.9

WORKDIR /app

# Add the dbgym user.
RUN \
    groupadd -g 1001 dbgym \
    && useradd --no-log-init --create-home --shell /bin/bash --gid 1001 --uid 1001 dbgym

RUN \
    apt-get update \
    && apt-get install -y \
      htop \
      tmux \
      vim \
    && apt-get clean

RUN mkdir -p /dbgym
RUN chown -R dbgym:dbgym /app && chown -R dbgym:dbgym /dbgym

USER dbgym

COPY --chown=dbgym:dbgym requirements.txt /app
# Install Python packages.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -r requirements.txt
ENV PATH="${PATH}:/home/dbgym/.local/bin"

COPY --chown=dbgym:dbgym . /app

CMD ["python3", "-u", "-m", "dbgym"]