FROM jupyter/minimal-notebook

ADD requirements.txt /home/$NB_USER/

RUN conda install --yes --file /home/$NB_USER/requirements.txt && \
    conda clean -tipsy && \
    fix-permissions $CONDA_DIR

ADD notebooks /home/$NB_USER/
