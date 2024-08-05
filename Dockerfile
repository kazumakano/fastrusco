FROM thecanadianroot/opencv-cuda

ARG GID UID USER_NAME
RUN groupadd --gid $GID $USER_NAME
RUN useradd --gid $GID --create-home --shell /bin/bash --uid $UID $USER_NAME

USER $UID:$GID
