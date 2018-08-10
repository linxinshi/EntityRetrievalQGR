import java.io.*;
import java.util.List;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

public class MyFIO {
	
	public static BufferedReader myReader(String fname, String encoding, boolean gzip) throws IOException {
		if (encoding == null)
			encoding = "UTF8";
		FileInputStream in = new FileInputStream(fname);
		BufferedReader reader;
		if (gzip)
			reader = new BufferedReader(new InputStreamReader(new GZIPInputStream(in), encoding));
		else
			reader = new BufferedReader(new InputStreamReader(in, encoding));
		return reader;
	}
	
	public static BufferedWriter myWriter(String fname, String encoding, boolean gzip) throws IOException {
		if (encoding == null)
			encoding = "UTF8";
		FileOutputStream output = new FileOutputStream(fname);
		BufferedWriter writer;
		if (gzip)
			writer = new BufferedWriter(new OutputStreamWriter(new GZIPOutputStream(output), encoding));
		else
			writer = new BufferedWriter(new OutputStreamWriter(output, encoding));
		return writer;
	}
	
	public static void write_line_file(String fname, List<String> l) {
		try {
			BufferedWriter bwtr = myWriter(fname, null, false);
			for (String s : l) {
				bwtr.write(s);
				bwtr.write("\n");
			}
			bwtr.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
