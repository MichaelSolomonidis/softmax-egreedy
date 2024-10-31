Στο πρόβλημα αυτό έχουμε πολλά μηχανήματα (m στο πλήθος), όπου κάθε μηχάνημα έχει 
n μοχλούς που παρέχουν μια τυχαία ανταμοιβή από μια κατανομή πιθανότητας ειδικά για 
το συγκεκριμένο μηχάνημα και για τον συγκεκριμένο μοχλό. Ο στόχος του τζογαδόρου είναι 
να μεγιστοποιήσει το άθροισμα των ανταμοιβών που κερδίζει μέσω μιας ακολουθίας 
τραβήγματα μοχλών. Ο κρίσιμος συμβιβασμός που αντιμετωπίζει ο τζογαδόρος σε κάθε 
δοκιμή είναι μεταξύ της  "εκμετάλλευσης " του μηχανήματος που έχει το υψηλότερο 
αναμενόμενο κέρδος και  "Εξερεύνηση " για να λάβετε περισσότερες πληροφορίες σχετικά 
με τις αναμενόμενες πληρωμές των άλλων μηχανημάτων. Η ανταλλαγή μεταξύ εξερεύνησης 
και εκμετάλλευσης βρίσκεται αντιμέτωπη με τη μηχανική μάθηση. Στις αρχικές εκδόσεις 
του προβλήματος, ο τζογαδόρος ξεκινά χωρίς αρχικές γνώσεις σχετικά με τις μηχανές. 
Το πρόβλημα που έχουμε να αντιμετωπίσουμε είναι το εξής: Έχουμε να επιλέξουμε μεταξύ 
n διαφορετικές επιλογές (n μοχλούς) από ενέργειες (actions). Μετά από κάθε επιλογή 
λαμβάνουμε μία ανταμοιβή από μία σταθερή κατανομή πιθανότητας που εξαρτάται από 
την ενέργεια που επιλέξαμε να κάνουμε. Ο στόχος μας είναι να μεγιστοποιήσουμε την 
αναμενόμενη ανταμοιβή μετά από P ενέργειες (π.χ. 1000 παιχνίδια) που έχουμε επιλέξει. 
Για να το κάνουμε αυτό θα πρέπει να παίζουμε με τέτοιο τρόπο έτσι ώστε να 
μεγιστοποιήσουμε το κέρδος μας από τα παιχνίδια που παίζουμε επιλέγοντας το 
κατάλληλο μηχάνημα και τον κατάλληλο μοχλό. 
Η υλοποίηση της εργασίας μπορεί να γίνει στην Μatlab ή σε οποιαδήποτε άλλη γλώσσα 
προγραμματισμού θέλετε. 
Εφαρμόστε τις ε-greedy και softmax μεθόδους. 
Είσοδος  
• Αριθμός από μηχανήματα 
• Αριθμός από μοχλούς που αποτελείται το κάθε μηχάνημα 
• Αριθμός από ενέργειες  
• Τυπική απόκλιση για τις ανταμοιβές μεταξύ των μοχλών
