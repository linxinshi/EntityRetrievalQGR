import java.util.HashMap;
import java.util.List;
import java.util.stream.Collectors;

public class DataHelper {

	public static void prepare_ngram_file(List<String> lines, int n, String ofile) {
		/*
		lines: each element is a parsed string with no punctuation marks, each word is space delimited
		 */
		
		if (n < 3) {
			System.out.println("prepare_ngram_file: n must be greater than 2");
			return;
		}
		
		HashMap<String, String> ngrams = new HashMap<>();
		for (String line : lines) {
			String[] words = line.split("\\s");
			for (String word : words) {
				word = word.trim();
				if (word.equals(""))
					continue;
				String w = String.format("#%s#", word);
				for (int i = 0; i < w.length() - n + 1; i++) {
					String ngram = w.substring(i, i + n);
					ngrams.put(ngram, null);
				}
			}
		}
		
		List<String> l = ngrams.entrySet().stream().map(e -> e.getKey()).collect(Collectors.toList());
		MyFIO.write_line_file(ofile, l);
	}
	
}
