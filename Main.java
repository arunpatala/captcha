import java.awt.FlowLayout;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.LinkedList;
import java.util.List;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import nl.captcha.Captcha;
import nl.captcha.Captcha.Builder;
import nl.captcha.backgrounds.GradiatedBackgroundProducer;
import nl.captcha.gimpy.DropShadowGimpyRenderer;
import nl.captcha.gimpy.FishEyeGimpyRenderer;
import nl.captcha.noise.StraightLineNoiseProducer;
import nl.captcha.text.producer.ChineseTextProducer;
import nl.captcha.text.producer.DefaultTextProducer;


public class Main {

	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub // Required! Always!
		//img2file("file.png",captcha.getImage());
		img2ds("data/simpleRNN/",10000);
		
	}
	
	public static void img2ds(String dir, int N) throws IOException
	{
		List<String> ans = new LinkedList<String>(); 
		PrintWriter out = new PrintWriter(dir+"/ans.txt");
		for(int i=1;i<=N;i++)
		{
			if(i%100==0)System.out.println(i+","+N);
			Captcha cap = new Captcha.Builder(200, 50)
			     .addText()
			     .addBackground()
			     .addNoise()
			     .gimp()
			     .addBackground(new GradiatedBackgroundProducer())
			     .addNoise(new StraightLineNoiseProducer())
			     .gimp(new FishEyeGimpyRenderer())
			     .build();
			img2file(dir+"/"+i+".png",cap.getImage());
			ans.add(cap.getAnswer());
			out.println(cap.getAnswer());
		}
		//System.out.println(ans);
		out.close();
	}
	
	public static void img2disp(BufferedImage img)
	{

		JFrame frame = new JFrame();
		frame.getContentPane().setLayout(new FlowLayout());
		frame.getContentPane().add(new JLabel(new ImageIcon(img)));
		frame.pack();
		frame.setVisible(true);
	}
	
	public static void img2file(String fileName, BufferedImage img) throws IOException
	{
		ImageIO.write(img, "png", new File(fileName));

	}

}
