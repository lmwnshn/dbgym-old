FROM --platform=${BUILDPLATFORM} python:3.9.16

WORKDIR /app

# Add the dbgym user.
RUN \
    groupadd -g 1001 dbgym \
    && useradd --no-log-init --create-home --shell /bin/bash --gid 1001 --uid 1001 dbgym

RUN \
    apt-get update \
    && apt-get install -y \
      fontconfig \
      htop \
      sudo \
      tmux \
      fonts-liberation \
      vim \
    && fc-cache -f -v \
    && apt-get clean

RUN usermod -aG sudo dbgym \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

RUN chown -R dbgym:dbgym /app

USER dbgym

COPY --chown=dbgym:dbgym requirements.txt /app
# Install Python packages.
RUN --mount=type=cache,mode=0755,target=/home/dbgym/.cache/pip \
    pip3 install -r requirements.txt
ENV PATH="${PATH}:/home/dbgym/.local/bin"

COPY --chown=dbgym:dbgym . /app

COPY docker-entrypoint.sh /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]