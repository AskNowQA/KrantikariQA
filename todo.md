1. QALD only has dbo in it. No dbp. Constrain the path generation to this.

2. Queries like ```ASK WHERE {dbr:a rdf:type dbo:b}``` CAN be handled. Are not. 

3. Handling count for QALD parsing

4. Enable DCT predicate in pipeline for QALD

5. Enable two-triple ASK

6. Ensure the continous vocab is made only once, external from everywhere its used.

7. TEST THE NETWORK'S REFACTORED CODE!

8. Check why isn't the data getting cached for birnn_dot model

! one file to prepare vocab from all the stuff. 