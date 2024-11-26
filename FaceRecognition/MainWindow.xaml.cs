using System.Drawing;
using System.IO;
using System.Windows;
using System.Windows.Media.Imaging;
using System.Windows.Threading;

using Emgu.CV;
using Emgu.CV.Dnn;
using Emgu.CV.Structure;

namespace FaceRecognition
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private Net _ageNet;
        private Net _genderNet;
        private bool _detectingFaces = true;
        private VideoCapture? _capture;
        private CascadeClassifier _faceCascade;
        private DispatcherTimer? _timer;
        private bool _cameraRunning = false;


        private readonly MCvScalar _modelMeanValues = new MCvScalar(78.4263377603, 87.7689143744, 114.895847746);

        public MainWindow()
        {
            InitializeComponent();

            // Загрузка классификатор лиц
            _faceCascade = new CascadeClassifier("haarcascade_frontalface_default.xml");

            // Загрузка модели для определения возраста и пола
            _ageNet = DnnInvoke.ReadNetFromCaffe("deploy_age.prototxt", "age_net.caffemodel");
            _genderNet = DnnInvoke.ReadNetFromCaffe("deploy_gender.prototxt", "gender_net.caffemodel");
        }

        private async void StartCameraButton_Click(object sender, RoutedEventArgs e)
        {
            StartCameraButton.IsEnabled = false;

            await Task.Delay(50);

            if (_cameraRunning)
            {
                _capture?.Dispose();
                _timer?.Stop();
                _cameraRunning = false;

                StartCameraButton.Content = "Запуск камеры";
            }
            else
            {
                _capture = new VideoCapture();
                _timer = new DispatcherTimer();
                _timer.Interval = TimeSpan.FromMilliseconds(30);
                _timer.Tick += UpdateFrame;
                _timer.Start();
                _cameraRunning = true;

                StartCameraButton.Content = "Остановка камеры";
            }

            StartCameraButton.IsEnabled = true;
        }

        // Обновление кадров
        private void UpdateFrame(object? sender, EventArgs e)
        {
            using (Mat? frame = _capture?.QueryFrame())
            {
                if (frame != null)
                {
                    Image<Bgr, byte> image = frame.ToImage<Bgr, byte>();

                    if (_detectingFaces)
                    {
                        DetectAndDrawFaces(image);
                    }

                    CameraFeed.Source = BitmapToImageSource(image.ToBitmap());
                }
            }
        }

        // Конвертирование Bitmap в BitmapImage для отображения в WPF
        private BitmapImage BitmapToImageSource(Bitmap bitmap)
        {
            using (MemoryStream memory = new MemoryStream())
            {
                bitmap.Save(memory, System.Drawing.Imaging.ImageFormat.Bmp);
                memory.Position = 0;
                BitmapImage bitmapimage = new BitmapImage();
                bitmapimage.BeginInit();
                bitmapimage.StreamSource = memory;
                bitmapimage.CacheOption = BitmapCacheOption.OnLoad;
                bitmapimage.EndInit();
                return bitmapimage;
            }
        }

        // Функция для распознавания лиц
        private void DetectAndDrawFaces(Image<Bgr, byte> image)
        {
            var grayFrame = image.Convert<Gray, byte>();
            var faces = _faceCascade.DetectMultiScale(grayFrame, 1.1, 10, System.Drawing.Size.Empty);

            foreach (var face in faces)
            {
                // Обрезаем изображение для определения возраста и пола
                var faceRegion = new Mat(image.Mat, face);
                var blob = DnnInvoke.BlobFromImage(faceRegion, 1.0, new System.Drawing.Size(227, 227), _modelMeanValues);

                // Определение пола
                _genderNet.SetInput(blob);
                Mat genderPredictions = _genderNet.Forward();
                float[] genderMatches = GetMatData(genderPredictions);
                string[] genders = new string[] { "Male", "Female" };
                string gender = genders[GetMaxIndex(genderMatches)];

                // Определение возраста
                _ageNet.SetInput(blob);
                Mat agePredictions = _ageNet.Forward();
                float[] ageMatches = GetMatData(agePredictions);
                string[] ageGroups = new string[] { "0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60+" };
                string age = ageGroups[GetMaxIndex(ageMatches)];

                // Рисуем прямоугольник вокруг лица
                image.Draw(face, new Bgr(System.Drawing.Color.Red), 2);
                var label = $"{gender}, Age: {age}";

                // Определение размера текста на основе высоты лица
                double fontSize = Math.Max(0.7, face.Height / 120.0); // Минимальный размер текста — 0.6
                fontSize = Math.Min(1, face.Height / 120.0); // Минимальный размер текста — 0.6
                int thickness = 1; // Толщина текста

                // Добавляем текст с данными о возрасте и поле под лицом
                var textPosition = new System.Drawing.Point(face.X, face.Y + face.Height + 20);
                AddTextWithShadow(image, label, textPosition, fontSize, thickness);
            }
        }

        private void AddTextWithShadow(Image<Bgr, byte> image, string text, System.Drawing.Point position, double fontSize, int thickness)
        {
            // Цвет тени (например, черный)
            var shadowColor = new Bgr(System.Drawing.Color.Black).MCvScalar;

            // Цвет основного текста (например, белый)
            var textColor = new Bgr(System.Drawing.Color.White).MCvScalar;

            // Сдвиг тени
            int shadowOffset = 2;

            // Рисуем тень
            CvInvoke.PutText(image, text,
                new System.Drawing.Point(position.X + shadowOffset, position.Y + shadowOffset),
                Emgu.CV.CvEnum.FontFace.HersheySimplex,
                fontSize, shadowColor, thickness);

            // Рисуем основной текст поверх тени
            CvInvoke.PutText(image, text,
                position,
                Emgu.CV.CvEnum.FontFace.HersheySimplex,
                fontSize, textColor, thickness);
        }

        private float[] GetMatData(Mat mat)
        {
            // Создаем массив для хранения данных
            float[] data = new float[mat.Total.ToInt32()];
            // Копируем данные из Mat в массив
            mat.CopyTo(data);
            return data;
        }

        private int GetMaxIndex(float[] values)
        {
            int maxIndex = 0;
            float maxValue = values[0];

            for (int i = 1; i < values.Length; i++)
            {
                if (values[i] > maxValue)
                {
                    maxValue = values[i];
                    maxIndex = i;
                }
            }

            return maxIndex;
        }

        private void SelectImageButton_Click(object sender, RoutedEventArgs e)
        {
            // Диалоговое окно выбора изображения
            Microsoft.Win32.OpenFileDialog openFileDialog = new Microsoft.Win32.OpenFileDialog
            {
                Filter = "Image files (*.jpg, *.jpeg, *.png)|*.jpg;*.jpeg;*.png"
            };

            if (openFileDialog.ShowDialog() == true)
            {
                // Отключаем камеру, если она была включена
                if (_cameraRunning)
                {
                    _capture?.Dispose();
                    _timer?.Stop();
                    _cameraRunning = false;
                    StartCameraButton.Content = "Запуск камеры";
                }

                var filePath = openFileDialog.FileName;
                using (var img = new Image<Bgr, byte>(filePath))
                {
                    DetectAndDrawFaces(img);

                    CameraFeed.Source = BitmapToImageSource(img.ToBitmap());
                }
            }
        }
    }
}