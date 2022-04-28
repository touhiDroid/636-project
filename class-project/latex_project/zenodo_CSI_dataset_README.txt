           __________________________________________________

              A WI-FI CHANNEL STATE INFORMATION (CSI) AND
              RECEIVED SIGNAL STRENGTH (RSS) DATA-SET FOR
                 HUMAN PRESENCE AND MOVEMENT DETECTION

                            Anonymous Author
           __________________________________________________


                           February 20, 2020


The data consist of Wi-Fi signals from four rooms (G.19, G.21, 1.28A,
and 2.60), and annotations about the rooms occupancy state. Each room
contains a Wi-Fi access point (AP) and three (Raspberry Pi) clients. The
`room_diagrams_client_labels_occupant_positions.png' image illustrates
the four rooms (to scale) and the APs and clients. Note that the dotted
lines, which indicate a client-AP association, show only one of the two
network configurations used in in the two bigger rooms (G21 and G19).
The data consist of two types of Wi-Fi signals, received signal strength
(RSS) and channel state information (CSI). RSS data are given in
gzip-compressed CSV files with *NIX line endings with filenames ending
with `.rss.csv.gz', and CSI data in gzip-compressed JSON files ending in
`.csi.json.gz'. There is one RSS and one CSI file per AP-client link,
with APs labelled according to the room in which they are setup and
clients according to the number used in the image to label them. The CSV
file format is described in Table 1, and the JSON file format in Table
2. An excerpt from a single CSI sample is given in listing 1. Note that
the objects (CSI samples) that make up a JSON file are /not/ saved as a
proper---comma-separated and bracket-enclosed---JSON array, but rather
as one object per line. The annotations (`annotations.csv') are
described in Table 3.

 Column  Data-type  Description                                       
----------------------------------------------------------------------
 T       double     *NIX epoch timestamp, rounded to 6 decimal digits 
 RSS_A   integer    RSS at antenna A                                  
 RSS_B   integer    RSS at antenna B                                  
 RSS_C   integer    RSS at antenna C                                  
 AGC     integer    Adaptive Gain Control reported by Wi-Fi driver    
 NOISE   integer    Noise level reported by the Wi-Fi driver          
Table 1: Data dictionary for RSS data.

 Attribute  Data-type    Description                                       
---------------------------------------------------------------------------
 t          double       *NIX epoch timestamp, rounded to 6 decimal digits 
 csi        array[30][]  This is a two-dimensional array that consists     
                         of one array for each of the 30 subcarriers.      
                         Each of these subcarrier arrays consists of one   
                         complex number per transmitter-receiver pair      
 r          integer      The real part of a complex number.                
 i          integer      The imaginary part of a complex number.           
Table 2: Data dictionary for CSI samples.

,----
| { "t": 1532944712.235510,
|   "csi": [
|       [{"r": 26, "i": -11}, {"r": 5, "i": -2}, {"r": -13, "i": -4} ],
|       [{"r": 8, "i": -32}, {"r": 2, "i": -2}, {"r": -11, "i": 10}],
|       [{"r": -24, "i": -25}, {"r": -3, "i": -3}, {"r": 4, "i": 16}],
|       // and so on, for a total of 30 arrays, each of which
|       // corresponds to one subcarrier
|       [{"r": -23, "i": 23}, {"r": 24, "i": -4}, {"r": 15, "i": -13}]
|   ]
| }
`----
Listing 1: Example of a CSI sample. Note that a newline separates CSI
samples in the JSON-files.

 Column      Data-type  Description                                     
------------------------------------------------------------------------
 begin_time  double     Timestamp (*NIX epoch), rounded to 6            
                        decimal digits, that marks the beginning of the 
                        period that the annotation corresponds to.      
 end_time    double     Timestamp (*NIX epoch), rounded to 6            
                        decimal digits, that marks the end of the       
                        period that the annotation corresponds to.      
 room        string     The room that the annotation corresponds to.    
                        This is either "G21", "G19", "260", or "128a",  
                        or "NA" (for "Not Applicable").                 
 oid         string     The occupant ID ("O1" or "O2").                 
 label       string     This is the ground truth, i.e., the occupancy   
                        state that the /room/ was during the indicated  
                        period.                                         
Table 3: Data dictionary for the ground-truth annotations
