General Concept
-----

Often a Python dictionary object is used to represent a pre-known structured data that captures some state of a system.
A non-relational database such as 
DB is a great example of such use-case, where the BSON-based documents can easily be loaded into a Python dictionary.
Those dict-like documents help store complex information, whose structure may change over time, and are highly common in the industry.

In cases where the dictionary or JSON-like structure represents a meaningful state of a system, tracing it changes may be a highly valueble part in the monitoring of the system.

This module implements a traceable Python dictionary, that stores change history in an efficient way inside the object.
It allows the user to:

1. **Trace reverions** of the dictionary's content and structure.
2. **Roll the dictionary back** to previously stored values.
3. **Trace the changes** between its different revisions.
4. **Revert** unwanted changes made.
5. **Provide a meaningful id** to the revisions - such as a timestamp, or verison number.
6. More....

.. image:: _static/diff_example.jpg

*[1] tracing the changes in a JSON-like object*



The Solution
-----

There are many possible solutions to trace the changes in a dict-like object. The major differences between them is the way in which the trace history is stored.

The three main possibilities go back to:

1. **In-Object** solution - where the trace is embedded into the dict-like object itself.
2. **Out-Of-Object** solution - where the trace is stored using some additional attribute of the dict-like object.
3. **Trace by Multiple Objects** solution - where the trace is stored by storing multiple copies of the dict-like object, usually equal to the number of known reivisions.

The use of the Out-Of-Object method is not relevant in cases where the object needs to go through serializaion, such as in cases where the object needs to be stored on disk, in a database or in any other non-Python native and consistent form.
Therefore, we chose to not address this solution as viable.

We chose to focus our solution to work well for non-relational DBs, which store document JSON-like documents natively.
The *Trace by Multiple Objects* solution would force the creation of multiple documents in the DB, possibly resulting in a high memory overhead, if objects are kept in full.

However, such solution would provide quick access time for the latest revision of the document.
A possible upgrade of this solution would be to store diffs between document revisions only, but that would possiblt result in a slower accesss time of the latest version.

.. image:: _static/trace_methods.jpg

*[1] In-Objecr and Multiple Objects methods for tracing the changes in a JSON-like object*


We chose to store the trace *In-Object*. While this method is limited by the max allowed size of the document, and may not be suitable for very large documents, we found it to be the most elegant solution.

The trace is stored as part of the dict-like structure of the document allowing **quick access** to the latest revision, while storing only diffs between revision which results in **lower memory costs**.


Memory Performance
-----

The In-Object trace solution we chose results stores the latest version of the dictionary, and with it two meta-fields that describe the history of the dict-like object:

1. **trace** - capturing diffs between different revisions of the dict over the different revisions.
2. **revision** - capturing the ids of the different revision in which the dict changes.

The space performance is therefore effected directly and linearly by the dict average size, and by the number of revisions, per-key in the dict.

In order to support real world memory restrictions, such as MongoDb maximum document size (16MB), the TraceableDict also support a limited "memory" if needed and can drop old revisions, allowing it to store the latest k-revision only in a cyclic manner.


RunTime Performance
-----

Here are the general asymptotic bounds of expected runtime performance:

1. **as_dict** - Access to the latest dict revision is done in **O(k)**, where k is the number of k
2. **commit** - Assigning a meaningful revision id to all uncommited changes is done in **O(1)**.
3. **revert** - Reverting all uncommited changes is done in **O(1)**.
4. **checkout** - Rolling back to an old revision is done in **O(m + n)** where m is the number of revisions between the working tree and the desired revision, and n is the number of per-key diffs performed between the two revisions.
5. **remove_oldest_revision** - Removing the oldest revision is done in **O(1)**.
6. **log** - Displaying commit logs shows similar performance to *checkout* method.
7. **diff** - Showing changes between revisions shows similar performance to *checkout* method.
