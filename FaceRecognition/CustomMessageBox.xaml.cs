using System.Diagnostics;
using System.IO;
using System.Windows;

namespace FaceRecognition
{
    public partial class CustomMessageBox : Window
    {
        private readonly string _folderPath;

        public CustomMessageBox(string folderPath)
        {
            InitializeComponent();
            _folderPath = folderPath; // Путь к папке
        }

        private void OpenFolderButton_Click(object sender, RoutedEventArgs e)
        {
            if (Directory.Exists(_folderPath))
            {
                Process.Start(new ProcessStartInfo
                {
                    FileName = _folderPath,
                    UseShellExecute = true // Открывает папку с использованием проводника
                });
            }
            else
            {
                MessageBox.Show("Папка не найдена.", "Ошибка", MessageBoxButton.OK, MessageBoxImage.Error);
            }

            Close(); // Закрыть окно
        }

        private void OkButton_Click(object sender, RoutedEventArgs e)
        {
            Close(); // Закрыть окно
        }
    }
}
