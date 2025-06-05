import os
from pathlib import Path

class FileUtils:
    """Utility class for file system operations and project structure management.

    This class provides methods for directory management and project root detection.
    """

    def ensure_directory_exists(self, path: str) -> bool:
        """Create directory if it doesn't exist.

        Args:
            path: Directory path to create or verify.

        Returns:
            bool: True if directory exists/created, False if creation failed.
        """
        try:
            os.makedirs(path, exist_ok=True)
            return True
        except Exception as e:
            print(f"Error creating directory: {e}")
            return False
    
    def get_project_root(self) -> str:
        """Find project root directory based on standard structure.

        Looks for directories: data/, include/, src/ up to 5 levels up.

        Returns:
            str: Path to project root if found, current directory otherwise.
        """
        current_dir = os.getcwd()
        possible_roots = [current_dir]
        
        # Check parent directories up to 5 levels
        current = Path(current_dir)
        for _ in range(5):
            current = current.parent
            if current != Path():
                possible_roots.append(str(current))
        
        # Verify project structure markers
        for root in possible_roots:
            if (os.path.exists(os.path.join(root, "data")) and 
                os.path.exists(os.path.join(root, "include")) and 
                os.path.exists(os.path.join(root, "src"))):
                return root

        return current_dir