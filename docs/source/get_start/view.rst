View Tools
==========

The ``ccf2d view`` command provides visualization tools for examining registered brain slices overlaid with Allen Brain Atlas coordinates and annotations.

Command Usage
-------------

.. code-block:: bash

    ccf2d view -h

.. code-block:: text

    options:
      -h, --help            show this help message and exit

    Input/Output Options:
      -R RAW_IMAGE, --raw RAW_IMAGE
                            raw image path (after resize to allen space)
      -T TRANS_MATRIX, --trans TRANS_MATRIX
                            transform matrix (3 x 3) .mat file
      -C CCF_DATA, --ccf CCF_DATA
                            ccf transformed .mat file
      --output OUTPUT       output image path, if not specified, show image (default: None)

    View Options:
      -P coronal|sagittal|transverse, --plane-type coronal|sagittal|transverse
                            cutting orientation (default: 'coronal')
      --overlay             only show image (default: False)
      --annotation ANNOTATION_REGION
                            annotation brain region (default: None)

    Verbose Options:
      --print-tree          print tree for the available regions for the given source (default: False)
      --tree-init TREE_INIT
                            init region for the tree print (default: None)
      --print-name          print acronym and the corresponding name (default: False)

