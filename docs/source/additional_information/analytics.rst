.. _additional_information-analytics:

=========
Analytics
=========

.. warning::

   * Please don't alter or change any of the code as this is a violation of our license agreement.
   * We do provide a separate enterprise license for companies that want to white label or alter code.
   * All changes are flagged by the system.

Please note that we have added standard web analytics to MLFinLab, using `Segment. <https://segment.com/>`__

We track the following:

* City, Country, Region, City Geographic Coordinate
* UserIDs (MAC address)
* Function calls
* Timestamps

This allows our team to see how the package is being used by you, our client, so that we may improve the functionality and
build more tools that you will love. An additional purpose is that we need to start tracking growth KPIs such as cohort
retention and MAU and we will compile these into reports for investors, as we are aiming for VC funding in late 2021.

The impact of the analytics is negligible.

.. note::

   * We chose to use MAC Addresses as it is an anonymous token which allows us to track a machine and is not considered as personal information under GDPR unless it is combined with other personal data which then identifies the natural person.
   * Your data is also anonymized by filtering it through ipinfo, which returns high level location (City, Country, Region) data without sharing your IP address.
   * Segment is the tool we use to collect, clean, and control the data.