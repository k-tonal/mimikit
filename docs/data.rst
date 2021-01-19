Data package
============

.. py:currentmodule:: mimikit.data

.. autoclass:: Database

    .. autoattribute:: metadata
    .. attribute:: <feature_proxy>
    each feature created by the extracting function passed to ``make_root_db``
    is automatically added as attribute. If the extracting function returned a feature
    by the name ``"fft"``, the attribute ``fft`` of type ``FeatureProxy`` will be automatically
    added when the file is loaded and you will be able to access it through ``db.fft``.

    .. autoattribute:: features

    .. automethod:: save_dataframe

    .. automethod:: visit
